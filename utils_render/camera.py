
import glm
import numpy as np
import torch
import torch.nn.functional as F
PI = 3.1415926536

class Camera:
    def __init__(self, WIN_H, WIN_W, num_gpus):

        # draggin state
        self.is_dragging  = False
        self.is_panning   = False
        self.about_origin = False
        self.is_left_btn  = True
        self.is_rot       = True
        self.is_rot_z     = True
        self.fix_y = False

        self.drag_start       = torch.full([2], 0.0).share_memory_()
        self.drag_start_right  = torch.full([3], 0.0).share_memory_()
        self._drag_start_right  = torch.full([3], 0.0).share_memory_()
        
        self.drag_start_front  = torch.full([3], 0.0).share_memory_()
        self._drag_start_front = torch.full([3], 0.0).share_memory_()
        
        self.drag_start_down   = torch.full([3], 0.0).share_memory_()
        self.drag_start_center = torch.full([3], 0.0).share_memory_()
        self._drag_start_center = torch.full([3], 0.0).share_memory_()
        
        self.drag_start_origin = torch.full([3], 0.0).share_memory_()
        self.movement_speed = 1.5  # GUI move speed
        # self.movement_speed = 3  # GUI move speed

        # interal states
        self.width  = WIN_W
        self.height = WIN_H

        self.c2w        = torch.full([4,4], 0.0).share_memory_()
        self.w2c        = torch.full([4,4], 0.0).share_memory_()
        self.w2c_init   = torch.full([4,4], 0.0).share_memory_()
        # self.update_K(tar_K)
        
        # properties;
        self.center     = torch.full([3], 0.0).share_memory_() # camera's position
        self.v_front    = torch.full([3], 0.0).share_memory_()
        self.v_world_up = torch.full([3], 0.0).share_memory_()
        self.origin     = torch.tensor([0.0, 0.0, 0.0]).share_memory_() # original pos, target position.
        self.v_right    = torch.full([3], 0.0).share_memory_()
        self.v_down     = torch.full([3], 0.0).share_memory_()
        
        # labels for initialization.
        self.fin_initialization = torch.full([num_gpus], 0.0).share_memory_()

        # camera path control
        self.front_tck = None
        self.center_tck = None
        self.worldup_tck = None
        # this option should control whether current rotation (handled by right click) is controlled by the predefined camera path
        # loaded from the dataset, and interpolated with B-spline interpolation
        self.on_cam_path = False
        self.cam_path_u = 0.  # the parameter [0, 1] controlling camera path (from first camera interpolated to last one)
        self.num_gpus   = num_gpus

    def initialize_K_RT(self, tar_K, tar_RT, target):
        self.tar_RT = tar_RT; self.tar_K = tar_K
        self.reset_RT(tar_RT)

        self.origin *= 0;
        self.origin += target.cpu();
        
        # update RT matrix.
        self.update_trans()
        self.record_static_start_dir()
        self.w2c_init *= 0;
        self.w2c_init += self.w2c.clone();
    
    def reset_RT(self, tar_RT):
        tar_R_inv = torch.inverse( tar_RT[:3,:3] )
        worldup, front, center = self.calcualte_cam_data( tar_RT, tar_R_inv )
        # update camera's vectors in the shared memories.
        self.center     *= 0;
        self.center     += center;
        
        self.v_front    *= 0;
        self.v_front    += F.normalize( front[None] )[0];

        self.v_world_up *= 0;
        self.v_world_up += F.normalize( worldup[None] )[0];


    def set_fin_flag(self, rank):
        self.fin_initialization[rank] *= 0
        self.fin_initialization[rank] += 1.0

    def get_fin_flag(self):
        flag = 1.0
        for i in range(self.num_gpus):
            flag *= self.fin_initialization[i]

        return flag == 1.0
    
    def calcualte_cam_data(self, RT, R_inv):
        # up : -R_inv @ [0,1,0]; front : R_inv @ [0,0,1]
        T = RT[:3,-1]
        world_up, front, center = -R_inv[:3, 1], R_inv[:3, 2], -R_inv @ T

        return world_up.cpu(), front.cpu(), center.cpu()

    @property
    def has_cam_path(self):
        return self.front_tck is not None and self.center_tck is not None and self.worldup_tck is not None

    def update_trans(self):
        # update front vector;
        tmp_front = F.normalize( self.v_front[None] )[0]
        self.v_front *= 0; self.v_front += tmp_front;
        # update right vector;
        self.v_right *= 0; self.v_right += F.normalize( torch.cross( self.v_front, self.v_world_up )[None] )[0]
        # update down vector;
        self.v_down *= 0;  self.v_down += torch.cross(self.v_front, self.v_right)
        # update world up
        self.v_world_up *= 0; self.v_world_up += -self.v_down;

        self.c2w *= 0;
        self.c2w[-1,-1] += 1.0
        self.c2w[:3,0] += self.v_right
        self.c2w[:3,1] += self.v_down
        self.c2w[:3,2] += self.v_front
        self.c2w[:3,3] += self.center
        
        self.w2c *= 0;
        self.w2c += torch.inverse(self.c2w)

    def begin_drag(self, x, y, is_pan, about_origin, fix_y, rot_z):
        self.is_dragging = True
        
        self.drag_start *= 0; self.drag_start += torch.tensor([x, y]).float()
        self.drag_start_front *= 0; self.drag_start_front   += self.v_front;
        self.drag_start_right *= 0; self.drag_start_right   += self.v_right;
        self.drag_start_down  *= 0; self.drag_start_down    += self.v_down;
        self.drag_start_center *= 0; self.drag_start_center += self.center;
        self.drag_start_origin *= 0; self.drag_start_origin += self.origin;
        self.is_panning      = is_pan
        self.is_rot          = about_origin
        self.about_origin    = about_origin
        self.is_rot_z        = rot_z
        self.fix_y           = fix_y
        self.drag_cam_path_u = self.cam_path_u        

    def end_drag(self):
        self.is_dragging = False

    def drag_update_new(self, x, y, is_full_body=True, online_demo=False, only_rot_one_axis=False):
        if not self.is_dragging:
            return
        
        drag_curr = torch.tensor([x, y]).float()
        delta     = drag_curr - self.drag_start
        # if online_demo:
        #     delta    *= -1
        delta    *= self.movement_speed / max(self.height, self.width)

        if not is_full_body and (not online_demo):
            tmp_y = delta[1].clone(); delta[1] = delta[0]; delta[0] = tmp_y;
            delta[1] *= -1;
        
        if self.is_panning and (not self.is_rot): # moving, no problem here.
            diff = delta[1] * self.drag_start_right - delta[0] * self.drag_start_down
            self.center *= 0
            self.center += self.drag_start_center + diff

            if self.about_origin:
                self.origin *= 0
                self.origin += self.drag_start_origin + diff
                
        elif self.is_rot and (not self.is_panning):

            if self.fix_y:
                delta[1] = 0;

            m = glm.mat4(1)
            if only_rot_one_axis:
                if delta[1].abs() > delta[0].abs(): # only rotate the axis 1;
                    m = glm.rotate(m, -delta.numpy()[1] % 2 * PI, glm.vec3(self.drag_start_down.numpy()))
                else: # only rotate the axis 0;
                    m = glm.rotate(m, -delta.numpy()[0] % 2 * PI, glm.vec3(self.drag_start_right.numpy()))
            else: # rotation two axis at the same time.
                m = glm.rotate(m, -delta.numpy()[1] % 2 * PI, glm.vec3(self.drag_start_down.numpy()))
                m = glm.rotate(m, -delta.numpy()[0] % 2 * PI, glm.vec3(self.drag_start_right.numpy()))

            m = np.array(m)[:3,:3]
            
            self.v_front *= 0;
            self.v_front += torch.from_numpy( m @ self.drag_start_front.numpy() ).float()

            if self.about_origin:
                self.center *= 0
                self.center += -torch.from_numpy(m @ (self.origin - self.drag_start_center).numpy()) + self.origin
        
        elif self.is_rot_z and (not self.is_rot) and (not self.is_panning):
            m = glm.mat4(1)
            m = glm.rotate(m, delta.numpy()[1] % 2 * PI, glm.vec3(self.drag_start_front.numpy()))
            m = np.array(m)[:3,:3]

            self.v_world_up *= 0;
            self.v_world_up += torch.from_numpy( -m @ self.drag_start_down.numpy() ).float()

            if self.about_origin:
                self.center *= 0
                self.center += -torch.from_numpy(m @ (self.origin - self.drag_start_center).numpy()) + self.origin
            

        self.update_trans()

    def update_origin(self, origin):
        self.origin *= 0
        self.origin += origin.cpu()
    
    def record_static_start_dir(self):
        # record the started directions.
        self._drag_start_front  *= 0; self._drag_start_front  += self.v_front;
        self._drag_start_right  *= 0; self._drag_start_right  += self.v_right;
        self._drag_start_center *= 0; self._drag_start_center += self.center;

    def yaw(self, angle, is_full_body, online_demo=False):
        # yaw along the right axis, calculate the rotation matrix.
        m = glm.mat4(1)
        if online_demo:
            m = glm.rotate(m, -glm.radians(angle) % 2 * PI, glm.vec3(self._drag_start_right.numpy()))
        else:
            if not is_full_body:
                m = glm.rotate(m, glm.radians(angle) % 2 * PI, glm.vec3(self.v_world_up.numpy()))
            else:
                m = glm.rotate(m, -glm.radians(angle) % 2 * PI, glm.vec3(self._drag_start_right.numpy()))
        
        m = np.array(m)[:3,:3]

        self.v_front *= 0
        self.v_front += torch.from_numpy( m @ self._drag_start_front.numpy() ).float()

        self.center *= 0
        self.center += -torch.from_numpy(m @ (self.origin - self._drag_start_center).numpy()) + self.origin

        self.update_trans()

    def move(self, movement):
        xyz = movement * self.v_front
        delta = xyz * self.movement_speed
        self.center += delta
        if self.is_dragging:
            self.drag_start_center += delta
        self.update_trans()

    def get_c2w(self):
        # gives in numpy array (row major)
        return self.c2w[:3].float().cuda()

    def get_w2c(self):
        # gives in numpy array (row major)
        return self.w2c[:3].float().cuda()
    
    def get_w2c_init(self):
        return self.w2c_init[:3].float()