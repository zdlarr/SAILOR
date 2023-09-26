#include <math.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

// #include <omp.h>
#include "../include/cuda_helper.h"
#include "../include/octree.h"
// #include "./intersection.cpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <stdio.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

// #define _UI64_MAX 0xffffffffffffffffui64

#define UNDISTORT_COLOR  1

static inline int DivUp(int total, int grain) { return (total + grain - 1) / grain; }

void create_dir(std::string dir)
{

#if 0
	if (0 != access(dir.c_str(), 0))
	{
		// if this folder not exist, create a new one.
		int res = mkdir(dir.c_str());   // ���� 0 ��ʾ�����ɹ���-1 ��ʾʧ��
										   //���� ::_mkdir  ::_access Ҳ�У���֪��ʲô��˼
		if (res == -1)

			std::cout << "create path " << dir << " failed " << std::endl;

		else
			std::cout << "create path " << dir << " success " << std::endl;
	}
	else
		std::cout << dir << "Directory already exists." << std::endl;

#endif

}


#define BACK_CUT 1

__global__ void undistort_image_from_remap_kernel(
	int width,
	int height,
	int cam_num,
	int *d_list,
	float *dis,
	float *K,
	float2 *d_remap,
	float *d_depth_bg,
	ushort *d_depth,
	ushort *d_undis_depth,
	uchar3 *d_color,
	uchar3 *d_undis_color,
	bool is_depth)
{
	int u = blockDim.x * blockIdx.x + threadIdx.x;
	if (u >= width * height * cam_num) return;

	int row = u / width;

	int cnt = row / height;
	  
	float colf = d_remap[u].x;
	float rowf = d_remap[u].y;
	if (u == 428744)
		printf("%d %f %f \n", u, colf, rowf);


	int x_1 = floor(colf), y_1 = floor(rowf), x_2 = ceil(colf), y_2 = ceil(rowf);

	int off = cnt * width * height;
	if (is_depth)
	{
		d_undis_depth[u] = 0;
	}
	else
	{
		d_undis_color[u] = make_uchar3(0,0,0);

	}
	if (x_1 >= 0 && y_1 >= 0 && x_1 < width && y_1 < height
		&& x_2 >= 0 && y_2 >= 0 && x_2 < width && y_2 < height) {

		int pos_11 = y_1 * width + x_1,
			pos_21 = y_1 * width + x_2,
			pos_12 = y_2 * width + x_1,
			pos_22 = y_2 * width + x_2;

		float ratio_11 = 1. * (y_2 - rowf) * (x_2 - colf),
			ratio_21 = 1. * (y_2 - rowf) * (colf - x_1),
			ratio_12 = 1. * (rowf - y_1) * (x_2 - colf),
			ratio_22 = 1. * (rowf - y_1) * (colf - x_1);

		if (x_1 == x_2 && y_1 == y_2)
		{
			ratio_11 = 1.0f;
			ratio_12 = ratio_21 = ratio_22 = 0;
		}
		else if (y_1 == y_2)
		{
			ratio_11 = 1. * (x_2 - colf);
			ratio_21 = 1. * (colf - x_1);
		}
		else if (x_1 == x_2)
		{
			ratio_11 = 1. * (y_2 - rowf);
			float sum_depth = 0, sum_ratio = 0;

			if (d_depth[off + (pos_11)] != 0  && abs(d_depth[off + (pos_11)] - d_depth_bg[off + (pos_11)]) >= 40)
			{
				sum_ratio += ratio_11;
				sum_depth += ratio_11 * d_depth[off + (pos_11)];
			}
			if (d_depth[off + (pos_12)] != 0 && abs(d_depth[off + (pos_12)] - d_depth_bg[off + (pos_12)]) >= 40)
			{
				sum_ratio += ratio_12;
				sum_depth += ratio_12 * d_depth[off + (pos_12)];
			}
			if (d_depth[off + (pos_21)] != 0  && abs(d_depth[off + (pos_21)] - d_depth_bg[off + (pos_21)]) >= 40)
			{
				sum_ratio += ratio_21;
				sum_depth += ratio_21 * d_depth[off + (pos_21)];
			}
			if (d_depth[off + (pos_22)] != 0 && abs(d_depth[off + (pos_22)] - d_depth_bg[off + (pos_22)]) >= 40)
			{
				sum_ratio += ratio_22;
				sum_depth += ratio_22 * d_depth[off + (pos_22)];
			}
			/*	|| d_depth[off + (pos_21)] == 0 ||
				d_depth[off + (pos_12)] == 0 || d_depth[off + (pos_22)] == 0 ||
			 ||
				abs(d_depth[off + (pos_12)] - d_depth_bg[off + (pos_12)]) <= 5 ||
				abs(d_depth[off + (pos_21)] - d_depth_bg[off + (pos_21)]) <= 5 ||
				abs(d_depth[off + (pos_22)] - d_depth_bg[off + (pos_22)]) <= 5)
				return;*/
		/*	if (u < 10000)
				printf("%d %f %f\n", u, d_depth[off + (pos_11)], d_depth_bg[off + (pos_11)]);
		*/
		/*	if (u == 428744)
				printf(" %f %f \n",sum_depth, sum_ratio);
*/

			if(sum_ratio!= 0)
				d_undis_depth[u] = (ushort)(sum_depth / sum_ratio);

		/*	d_undis_depth[u] = (ushort)(ratio_11 * d_depth[off + (pos_11)] +
				ratio_21 * d_depth[off + (pos_21)] +
				ratio_12 * d_depth[off + (pos_12)] +
				ratio_22 * d_depth[off + (pos_22)]);*/
		}
		else
		{
			/*if (d_color[off + (pos_11)].x == 0 || d_color[off + (pos_21)].x == 0 ||
				d_color[off + (pos_12)].x == 0 || d_color[(off + pos_22)].x == 0)
				return;
*/
			float r = ratio_11 * d_color[off + (pos_11)].x +
				ratio_21 * d_color[off + (pos_21)].x +
				ratio_12 * d_color[off + (pos_12)].x +
				ratio_22 * d_color[off + (pos_22)].x;
			float g = ratio_11 * d_color[off + (pos_11)].y +
				ratio_21 * d_color[off + (pos_21)].y +
				ratio_12 * d_color[off + (pos_12)].y +
				ratio_22 * d_color[off + (pos_22)].y;
			float b = ratio_11 * d_color[off + (pos_11)].z +
				ratio_21 * d_color[off + (pos_21)].z +
				ratio_12 * d_color[off + (pos_12)].z +
				ratio_22 * d_color[off + (pos_22)].z;


			r = r > 0 ? (r < 256 ? r : 255) : 0;
			g = g > 0 ? (g < 256 ? g : 255) : 0;
			b = b > 0 ? (b < 256 ? b : 255) : 0;

			d_undis_color[u] = make_uchar3((uchar)r, (uchar)g, (uchar)b);

		}



	}
}

__global__ void get_remap_kernel(
	int width, 
	int height,
	int cam_num,
	int *d_list,
	float *dis,
	float *K,
	float2 *d_remap)
{
	int u = blockDim.x * blockIdx.x + threadIdx.x;
	if (u >= width * height * cam_num) return;

	int row = u / width;
	int col = u % width;


	int cnt = row / height;
	row = row - cnt * height;

	int cid = d_list[cnt];

	float rowf = row;
	float colf = col;

	//convert image to camera
	float f1 = K[(cid - 1 )* 5 + 0], f2 = K[(cid - 1) * 5 + 1], alpha = K[(cid - 1) * 5 + 4], u0 = K[(cid - 1) * 5 + 2], v0 = K[(cid - 1) * 5 + 3];
	
	float coory = (rowf - v0) / f2;
	float coorx = (colf - u0) / f1;
	coorx -= alpha * coory;

	float x1, y1, x2, y2, r2;
	x1 = coorx;
	y1 = coory;

	r2 = x1 * x1 + y1 * y1;
	x2 = x1 * (1 + dis[0] * r2 + dis[1] * r2*r2 + dis[4] * r2*r2*r2) + 2 * dis[2] * x1 * y1 + dis[3] * (r2 + 2 * x1 * x1);
	y2 = y1 * (1 + dis[0] * r2 + dis[1] * r2*r2 + dis[4] * r2*r2*r2) + dis[2] * (r2 + 2 * y1 * y1) + 2 * dis[3] * x1 * y1;


	coorx = x2;
	coory = y2;


	colf = coorx * f1 + f1*alpha*coory + u0;
	rowf = coory * f2 + v0;

	d_remap[u] = make_float2(colf, rowf);
}


void MultiLayerOctree::UndistortImage(std::string xml_file, std::string path, std::vector<std::string> list, std::string save_path)
{
	float depth_K[12][5], depth_dis[12][5], color_K[12][5], color_dis[12][5];
	// std::string xml_file = "./9kinect-matlab-0903-hyf_2k.xml";
	cv::FileStorage fs1(xml_file, cv::FileStorage::READ);
	if (!fs1.isOpened())
	{
		std::cout << "Open Config.xml Failed\n";
		return;
	}

	cv::FileNode fn;
	cv::FileNodeIterator fn_it;

	fn = fs1["CameraFxFyCxCy"];
	int cnt = 0;
	for (fn_it = fn.begin(); fn_it != fn.end(); fn_it++)
	{
		cv::Mat m;
		cv::read(*fn_it, m);
		memcpy(depth_K[cnt], m.data, 5 * sizeof(float));
		cnt++;
	}


	fn = fs1["CameraDis"];
	cnt = 0;
	for (fn_it = fn.begin(); fn_it != fn.end(); fn_it++)
	{
		cv::Mat m;
		cv::read(*fn_it, m);
		memcpy(depth_dis[cnt], m.data, 5 * sizeof(float));
		cnt++;
	}

	fn = fs1["CameraColorFxFyCxCy"];
	cnt = 0;
	for (fn_it = fn.begin(); fn_it != fn.end(); fn_it++)
	{
		cv::Mat m;
		cv::read(*fn_it, m);
		memcpy(color_K[cnt], m.data, 5 * sizeof(float));
		cnt++;
	}


	fn = fs1["CameraColorDis"];
	cnt = 0;
	for (fn_it = fn.begin(); fn_it != fn.end(); fn_it++)
	{
		cv::Mat m;
		cv::read(*fn_it, m);
		memcpy(color_dis[cnt], m.data, 5 * sizeof(float));
		cnt++;
	}
	fs1.release();
	

	std::vector<int> clist = { 1, 2, 3, 4, 5,6, 8, 9 };
	int cam_num = clist.size();
	// std::string path = "G:";
	
	//"C:/Users/admin/Desktop/KinectCapture_0221/KinectCapture/capture_image";// 
	 // ;// ;//C:/Users/Admin/Desktop/data0916/dzh/";//  "D:/data_dzh_03312/";// ;//"D:/data_dzh_03312/";//"E:/1221david/";
	//"22_0221_yxq","22_0117_jyw",
	
	// std::vector<std::string> list =
	// {
	// 	//"22_0414_data3","22_0414_data4",
	// 	"data4","data5" };//,"22_0414_data6" ,// "22_0402_zyk","22_0413_data2", "22_0413_data3" 
		//"22_0414_data7","22_0414_data8","22_0414_data2"  };// "22_0402_zyk","22_0413_data2", "22_0413_data3" 
 //, "22_0401_dz", "22_0401_sjr", "22_0401_data1", "22_0401_data2",
	//	"22_0401_data3","22_0401_data4","22_0401_data5" };//21_1106_zx","21_1112_ll","21_1115_hyf3","22_0223_data1","22_0223_data2",
	/*	"22_0223_data3",  "22_0117_llf", "22_0119_dz/dz", "22_0222_fxd", 
		"22_0222_ljf", "22_0222_ljf1", "22_0222_thy", "22_0223_hsy", "22_0223_xjm", 
		"22_0225_jyw1","22_0225_jyw2", "22_0225_jyw3"};

   */
	for (int li = 0; li < list.size(); li++)
	{

		thrust::host_vector<float> depth_bg(clist.size() * 1024 * 1024);

		char name_h[256];
		for (int i = 0; i < clist.size(); i++) {
			int id = clist[i];
			cv::Mat out_mat = cv::Mat::zeros(1024, 1024, CV_32FC1);
			cv::Mat js = cv::Mat::zeros(1024, 1024, CV_16UC1);
			for (int i = 1; i < 50; i++) {
				sprintf(name_h, "%s/%s/bg/depth/%d_%d.png", path.c_str(), list[li].c_str(), i, id);
				std::cout << "read " << name_h << std::endl;
				cv::Mat tmp = cv::imread(name_h, cv::IMREAD_ANYDEPTH);
				for (int y = 0; y < tmp.rows; y++) {
					for (int x = 0; x < tmp.cols; x++) {
						u_short d = tmp.at<u_short>(y, x);
						if (d > 0) {
							out_mat.at<float>(y, x) += 1.0 * d;
							js.at<u_short>(y, x)++;
						}
					}
				}
			}
			for (int y = 0; y < out_mat.rows; y++) {
				for (int x = 0; x < out_mat.cols; x++) {
					float d = out_mat.at<float>(y, x);
					int n = js.at<u_short>(y, x);
					if (n > 0) {
						out_mat.at<float>(y, x) = d / n;
					}
				}
			}

			memcpy(&depth_bg[i * 1024 * 1024], out_mat.data, 1024 * 1024 * sizeof(float));
			cv::Mat res;
			out_mat.convertTo(res, CV_16UC1);
			sprintf(name_h, "%s/%s/bg/avg_bg_%d.png", path.c_str(), list[li].c_str(), id);
			cv::imwrite(name_h, res);


		}
#if 0
		for (int i = 0; i < clist.size(); i++)
		{
			int id = clist[i];

			sprintf(name_h, "%s/%s/bg/avg_bg_%d.png", path.c_str(), list[li].c_str(), id);
			cv::Mat out_mat = cv::imread(name_h, cv::IMREAD_ANYDEPTH);

			out_mat.convertTo(out_mat, CV_32FC1);
			memcpy(&depth_bg[i * 1024 * 1024], out_mat.data, 1024 * 1024 * sizeof(float));
		 
		}
	
#endif
		thrust::device_vector<float> d_depth_bg = depth_bg;

		// std::string save_path = "G:/";// "C:/Users/admin/Desktop/tmp_kinect_data/";
		std::string save_dir = save_path + list[li] + "/undis_new";
		create_dir(save_dir);

		std::string color_dir = save_dir + "/color";
		create_dir(color_dir);



		int cur_frame = 0;
		std::string dir_name = list[li];

		std::vector<std::fstream> fs;
		fs.resize(cam_num);
		for (int i = 0; i < cam_num; i++)
		{
			std::string p = path + "/" + dir_name + "/time_ori_" + std::to_string(clist[i]) + ".txt";
			std::cout << p;
			fs[i].open(p, std::ios::in);
			if (fs[i].is_open())
			{
				std::cout << "file open ok\n";
			}
			else
			{
				p = path + "/" + dir_name + "/time_" + std::to_string(clist[i]) + ".txt";
				std::cout << p;
				fs[i].open(p, std::ios::in);
				std::cout << "file open error\n";
			}
		}


		thrust::device_vector<float>	d_depth_dis(12 * 5), d_depth_k(12 * 5), d_color_dis(12 * 5), d_color_k(12 * 5);
		cudaMemcpy(RAW_PTR(d_depth_dis), &(depth_dis[0][0]), 12 * 5 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(RAW_PTR(d_depth_k), &(depth_K[0][0]), 12 * 5 * sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(RAW_PTR(d_color_dis), &(color_dis[0][0]), 12 * 5 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(RAW_PTR(d_color_k), &(color_K[0][0]), 12 * 5 * sizeof(float), cudaMemcpyHostToDevice);

		thrust::device_vector<ushort>	d_ori_depth_pic(1024 * 1024 * cam_num, 0), d_depth_pic(1024 * 1024 * cam_num, 0);
		thrust::device_vector<uchar3>	d_ori_color_pic(1440 * 2560 * cam_num, make_uchar3(0, 0, 0)),
			d_color_pic(1440 * 2560 * cam_num, make_uchar3(0, 0, 0));
		thrust::device_vector<int> d_list(cam_num);
		cudaMemcpy(RAW_PTR(d_list), &(clist[0]), cam_num * sizeof(int), cudaMemcpyHostToDevice);

		int drows = 1024;
		int dcols = 1024;
		int rows = 1440;
		int cols = 2560;


		thrust::device_vector<float2> d_depth_remap(drows * dcols * cam_num);
		thrust::device_vector<float2> d_color_remap(rows * cols * cam_num);


		int block_num = 256;
		int grid_num = DivUp(drows * dcols * cam_num, block_num);
		get_remap_kernel << < grid_num, block_num >> > (
			dcols,
			drows,
			cam_num,
			RAW_PTR(d_list),
			RAW_PTR(d_depth_dis),
			RAW_PTR(d_depth_k),
			RAW_PTR(d_depth_remap)
			);
		cudaDeviceSynchronize();
		cudaGetLastError();
	/*	thrust::host_vector<float2> h_remap = d_depth_remap;
		for (int i = 0; i < 1; i++)
		{
			for (int c = 0; c < dcols; c++)
			{
				for (int r = 0; r < drows; r++)
					printf("%d %d %f %f \n", r, c, h_remap[r * dcols + c].x, h_remap[r * dcols + c].x);
			}
		}*/

		block_num = 256;
		grid_num = DivUp(rows * cols * cam_num, block_num);
		get_remap_kernel << < grid_num, block_num >> > (
			cols,
			rows,
			cam_num,
			RAW_PTR(d_list),
			RAW_PTR(d_color_dis),
			RAW_PTR(d_color_k),
			RAW_PTR(d_color_remap)
			);
		cudaDeviceSynchronize();
		cudaGetLastError();


		thrust::host_vector<short> h_depth = d_depth_pic;
		thrust::host_vector<uchar3> h_color = d_color_pic;

#if 0

		d_depth_bg = depth_bg;

		std::string save_bg_dir = save_path + list[li] + "/bg/undis";
		create_dir(save_bg_dir);

		std::string color_bg_dir = save_bg_dir + "/color";
		create_dir(color_bg_dir);
		for (int f = 0; f < 50; f++)
		{
			for (int c = 0; c < cam_num; c++)
			{
				int cid = clist[c];
				std::string filename = path + "/" + list[li] + "/bg/depth/" + std::to_string(f) + "_" + std::to_string(cid) + ".png";
				//undistort depth
				cv::Mat dep_pic = cv::imread(filename, cv::IMREAD_ANYDEPTH);// = undistort(filename, &(depth_dis[cid - 1][0]), &(depth_K[cid - 1][0]), true);
				checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(&(d_ori_depth_pic[drows * dcols * c])), dep_pic.data, drows * dcols * sizeof(short), cudaMemcpyHostToDevice));
			}

			for (int c = 0; c < cam_num; c++)
			{

				int cid = clist[c];
				std::string filename = path + "/" + list[li] + "/bg/pic/" + std::to_string(f) + "_" + std::to_string(cid) + ".jpg";
				cv::Mat color_pic = cv::imread(filename, cv::IMREAD_COLOR);
				checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(&(d_ori_color_pic[rows * cols * c])), color_pic.data, rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice));
			}




			int block_num = 256;
			int grid_num = DivUp(drows * dcols * cam_num, block_num);
			undistort_image_from_remap_kernel << < grid_num, block_num >> > (
				dcols,
				drows,
				cam_num,
				RAW_PTR(d_list),
				RAW_PTR(d_depth_dis),
				RAW_PTR(d_depth_k),
				RAW_PTR(d_depth_remap),
				RAW_PTR(d_depth_bg),
				RAW_PTR(d_ori_depth_pic),
				RAW_PTR(d_depth_pic),
				nullptr,
				nullptr,
				true
				);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());

			block_num = 256;
			grid_num = DivUp(rows * cols * cam_num, block_num);
			undistort_image_from_remap_kernel << < grid_num, block_num >> > (
				cols,
				rows,
				cam_num,
				RAW_PTR(d_list),
				RAW_PTR(d_color_dis),
				RAW_PTR(d_color_k),
				RAW_PTR(d_color_remap),
				nullptr,
				nullptr,
				nullptr,
				RAW_PTR(d_ori_color_pic),
				RAW_PTR(d_color_pic),
				false
				);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());

			h_depth = d_depth_pic;
			h_color = d_color_pic;




#pragma omp parallel for
			for (int c = 0; c < cam_num; c++)
			{
				char rn[256];
				cv::Mat undis_depth(drows, dcols, CV_16UC1, &h_depth[c * drows * dcols]);
				sprintf(rn, "%s/depth_%05d.png", save_bg_dir.c_str(), cam_num * (cur_frame - 0) + c);
				cv::imwrite(rn, undis_depth);

			}
#pragma omp parallel for
			for (int c = 0; c < cam_num; c++)
			{
				char rn[256];

				cv::Mat undis_color(rows, cols, CV_8UC3, &h_color[c * rows * cols]);
				sprintf(rn, "%s/color/color_%05d.jpg", save_bg_dir.c_str(), cam_num * (cur_frame - 0) + c);
				cv::imwrite(rn, undis_color);
			}
			cur_frame++;
			std::cout << cur_frame << std::endl;
		}
		continue;
#endif

		uint64  depth_time;
		std::vector<int> cam_cnt(cam_num, 0);
		std::vector<bool> read_flag(cam_num, true);
		std::vector<int>	txt_frame(cam_num);
		std::vector<uint64> txt_color_time(cam_num);

		std::vector<uint64> txt_depth_time(cam_num);
		//#pragma omp parallel for
		uint64 min = 99999999999999;
		int min_index = -1;
		uint64 max = 0;
		bool finish_flag = false;

		while (!finish_flag)
		{
			min = 99999999999999;
			max = 0;
			for (int c = 0; c < cam_num; c++)
			{
				if (read_flag[c])
				{
					read_flag[c] = false;
					if (cam_cnt[c] < 5000)
					{
						fs[c] >> txt_frame[c] >> txt_color_time[c] >> txt_depth_time[c];
						printf("Read %s cam %d's frame %d      color time: %d, depth time: %d, delta time: %d\n", dir_name.c_str(), clist[c], txt_frame[c],
							txt_color_time[c], txt_depth_time[c], txt_depth_time[c] - txt_color_time[c]);
						cam_cnt[c]++;
					}
					else
					{
						finish_flag = true;
						break;
					}
				}
				if (min > txt_color_time[c])
				{
					min = txt_color_time[c];
					min_index = c;
				}
				if (max < txt_color_time[c])
					max = txt_color_time[c];
			}
			clock_t start, end;
			start = clock();
			if (max - min < 4000)
			{
				printf("\nsync new frame %d,  %d done\n", cur_frame, cam_num * cur_frame);
				int cnt = 0;
 
		/*		for (int c = 0; c < cam_num; c++)
				{
					printf("  cam: %d   ori fid: %d, color time: %d, depth time: %d, delta time: %d\n",
						clist[c], txt_frame[c], txt_color_time[c], txt_depth_time[c], txt_depth_time[c] - txt_color_time[c]);

					read_flag[c] = true;
				}if (li == 0 && cur_frame < 1255)
				{
					cur_frame++;
					continue;
				} */

				#pragma omp parallel for
				for (int c = 0; c < cam_num; c++)
				{
					printf("  cam: %d   ori fid: %d, color time: %d, depth time: %d, delta time: %d\n",
						clist[c], txt_frame[c], txt_color_time[c], txt_depth_time[c], txt_depth_time[c] - txt_color_time[c]);

					//std::cout << "  " << c << " ori frame id: " << txt_frame[c] << " color time: " << txt_color_time[c] << std::endl;
					read_flag[c] = true;
					char rn[256];
					int cid = clist[c];
					std::string filename = path + "/" + dir_name + "/depth/" + std::to_string(txt_frame[c]) + "_" + std::to_string(cid) + ".png";
					printf("Read depth %s\n", filename.c_str());
					//undistort depth
					cv::Mat dep_pic = cv::imread(filename, cv::IMREAD_ANYDEPTH);// = undistort(filename, &(depth_dis[cid - 1][0]), &(depth_K[cid - 1][0]), true);

					if (dep_pic.rows == 0)
					{
						printf("can't find depth image: %s\n", filename.c_str());
						dep_pic = cv::Mat(drows, dcols, CV_16UC1, 0);

					}
					if (dep_pic.rows != drows)
					{
						printf("depth pic rows error !!\n");
						//getchar();
					}

					cudaMemcpy(thrust::raw_pointer_cast(&(d_ori_depth_pic[drows * dcols * c])), dep_pic.data, drows * dcols * sizeof(short), cudaMemcpyHostToDevice);

 
				}

#if UNDISTORT_COLOR
#pragma omp parallel for
				for (int c = 0; c < cam_num; c++)
				{ 
					char rn[256];
					int cid = clist[c];
					std::string filename = path + "/" + dir_name + "/pic/" + std::to_string(txt_frame[c]) + "_" + std::to_string(cid) + ".jpg";
					printf("Read rgb %s\n", filename.c_str());
					cv::Mat color_pic = cv::imread(filename, cv::IMREAD_COLOR);

					if (color_pic.rows == 0)
					{
						printf("can't find depth image: %s\n", filename.c_str());
						color_pic = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

					}
					if (color_pic.rows != rows)
					{
						printf("depth pic rows error !!\n");
						//getchar();
					}
					cudaMemcpy(thrust::raw_pointer_cast(&(d_ori_color_pic[rows * cols * c])), color_pic.data, rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice);

				}
#endif
				end = clock();   //����ʱ��
				printf("read time cost : %f ms\n", double(end - start));
				start = clock();

				int block_num = 256;
				int grid_num = DivUp(drows * dcols * cam_num, block_num);
				undistort_image_from_remap_kernel << < grid_num, block_num >> > (
					dcols,
					drows,
					cam_num,
					RAW_PTR(d_list),
					RAW_PTR(d_depth_dis),
					RAW_PTR(d_depth_k),
					RAW_PTR(d_depth_remap),
					RAW_PTR(d_depth_bg),
					RAW_PTR(d_ori_depth_pic),
					RAW_PTR(d_depth_pic),
					nullptr,
					nullptr,
					true
					);
				cudaDeviceSynchronize();
				cudaGetLastError();
#if UNDISTORT_COLOR
				block_num = 256;
				grid_num = DivUp(rows * cols * cam_num, block_num);
				undistort_image_from_remap_kernel << < grid_num, block_num >> > (
					cols,
					rows,
					cam_num,
					RAW_PTR(d_list),
					RAW_PTR(d_color_dis),
					RAW_PTR(d_color_k),

					RAW_PTR(d_color_remap),
					nullptr,
					nullptr,
					nullptr,
					RAW_PTR(d_ori_color_pic),
					RAW_PTR(d_color_pic),
					false
					);
				cudaDeviceSynchronize();
				cudaGetLastError();
				h_color = d_color_pic;
#endif

				h_depth = d_depth_pic;

				end = clock();   //����ʱ��
				printf("remap time cost : %f ms\n", double(end - start));
				start = clock();


				#pragma omp parallel for
				for (int c = 0; c < cam_num; c++)
				{ 
					char rn[256];
					cv::Mat undis_depth(drows, dcols, CV_16UC1, &h_depth[c * drows * dcols]);
					sprintf(rn, "%s/depth_%05d.png", save_dir.c_str(), cam_num * (cur_frame - 0) + c);
					cv::imwrite(rn, undis_depth);
					std::cout << "write image " << rn << std::endl;
#if UNDISTORT_COLOR
					cv::Mat undis_color(rows, cols, CV_8UC3, &h_color[c * rows * cols]);
					sprintf(rn, "%s/color/color_%05d.jpg", save_dir.c_str(), cam_num * (cur_frame - 0) + c);
					cv::imwrite(rn, undis_color);
#endif
				}
//
				//#pragma omp parallel for

				//#pragma omp parallel for
				//for (int c = 0; c < cam_num; c++)
				//{
				//	/*char rn[256];
				//	 
				//*/
				//}
//#endif
				cur_frame++;
			}
			else
			{
				printf("drop camera %d , color time %f \n", clist[min_index], min);
				//std::cout << "drop camera  " << clist[min_index] << ", color time is " << min << std::endl;
				read_flag[min_index] = true;
			}
			end = clock();   //����ʱ��
			printf("save image time cost : %f ms\n", double(end - start)); 

		}
		for (int c = 0; c < cam_num; c++)
		{
			fs[c].close();// .open(p, std::ios::in);
		}
	}
}


/*
	Undistort color and depths from given intrinsic, distortion params.
*/

template <const uint32_t n_channels>
__global__ void build_from_remap_kernel(
	const float *src_img, // [2, h, w, 6] or [2,h,w,1]
	const float *distort, // [2, 5+3]
	const float *K, // [2, 3,3]
	// float *d_remap, // [2, h, w, 2]
	float * tar_img, // [2, h, w,6] or [2,h,w,1]
	const int num_views,
	const int height,
	const int width
) {
	#pragma unroll
    CUDA_KERNEL_LOOP(k, num_views*height*width) {
		uint32_t v_idx = (uint32_t) k / (height*width);
		uint32_t h_idx = (uint32_t) (k - v_idx*height*width) / width;
		uint32_t w_idx = (uint32_t) k - v_idx*height*width - h_idx*width;

		float f1 = K[v_idx*9 + 0], // width focal.
			  f2 = K[v_idx*9 + 4], // height focal.
			  alpha = 0.0f, // no scale transformations.
			  u0 = K[v_idx*9 + 2], // width  / 2.
			  v0 = K[v_idx*9 + 5]; // height / 2.

		int b_dix   = v_idx*8;
			// basic_remap_idx = v_idx*height*width*2+h_idx*width*2+w_idx*2;

		float coory = (h_idx - v0) / f2;
		float coorx = (w_idx - u0) / f1;
		coorx -= alpha * coory;
		// printf("coory %f, coorx %f \n", coory, coorx);

		float x1, y1, x2, y2, r2;
		x1 = coorx;
		y1 = coory;

		// printf("%f, %f, %f, %f, %f \n", distort[0+b_dix], distort[1+b_dix], distort[2+b_dix], distort[3+b_dix], distort[4+b_dix]);
		r2 = x1 * x1 + y1 * y1;
		float ws  = 1 + distort[0+b_dix]*r2 + distort[1+b_dix]*r2*r2 + distort[4+b_dix]*r2*r2*r2; // jingxiang.
		float ws2 = 1 + distort[5+b_dix]*r2 + distort[6+b_dix]*r2*r2 + distort[7+b_dix]*r2*r2*r2 + 1e-8; 
	 
		x2 = x1*ws/ws2 + 2*distort[2+b_dix]*x1*y1 + distort[3+b_dix]*(r2 + 2*x1*x1);
		y2 = y1*ws/ws2 +   distort[2+b_dix]*(r2 + 2*y1*y1) + 2*distort[3+b_dix]*x1*y1;
		
		coorx = x2;
		coory = y2;

		float w_f = coorx * f1 + f1 * alpha * coory + u0;
		float h_f = coory * f2 + v0;
	
		// d_remap[basic_remap_idx + 0] = colf;
		// d_remap[basic_remap_idx + 1] = rowf;
		// get the properties.

		int x_1 = (int) floor(w_f), y_1 = (int) floor(h_f);
		int x_2 = x_1 + 1, y_2 = y_1 + 1;
		// printf("x1  %d,x2 %d ,y1 %d, y2 %d\n", x_1, x_2, y_1, y_2);
		
		if (x_1 >= 0 && y_1 >= 0 && x_2 < width && y_2 < height) {
			float dx = w_f - (float) x_1;
			float dy = h_f - (float) y_1;
			float ws[4] = {0};
			ws[0] = (1 - dx) * (1 - dy),
			ws[1] = dx * (1 - dy),
			ws[2] = (1 - dx) * dy,
			ws[3] = dx * dy;
			
			#pragma unroll n_channels
			for (int i=0; i < n_channels; i++) {
				float vals[4] = {0}, weight_sum = 0.0;
				uint32_t basic_idx = v_idx*n_channels*height*width + i*height*width;
				vals[0] = src_img[basic_idx + y_1*width + x_1],
				vals[1] = src_img[basic_idx + y_1*width + x_2],
				vals[2] = src_img[basic_idx + y_2*width + x_1],
				vals[3] = src_img[basic_idx + y_2*width + x_2];

				#pragma unroll 4
				for (int j=0; j<4; j++) {
					if (vals[j] != 0) weight_sum += ws[j];
				}
				// printf("weight sum: %f\n", weight_sum);
				if (weight_sum == 0.0) tar_img[basic_idx + h_idx*width + w_idx] = 0.0;
				else {
					// weight sum of all the vals, the zero values are ignored.
					float val_final = (vals[0]*ws[0] + vals[1]*ws[1] + vals[2]*ws[2] + vals[3]*ws[3]) / weight_sum;
					// printf("val_final: %f\n", val_final);
					tar_img[basic_idx + h_idx*width + w_idx] = val_final;
				}
				
			}
		}
	}
}


std::vector<torch::Tensor> undistort_images(
	// torch::Tensor bgs, // [2, 3, h, w]
	torch::Tensor input_rgbs, // [2, 6, h, w]
	torch::Tensor input_depths, // [2, 1, h', w']
	// parameters of colors and depths.
	torch::Tensor K_colors, // [2, 3, 3]
	torch::Tensor K_depths, // [2, 3, 3]
	torch::Tensor DIS_C, // [2, 5+3]
	torch::Tensor DIS_D,  // [2, 5+3]
	const int device
) {
	cudaSetDevice(device); // on GPU device.
    CUDA_CHECK_ERRORS();

	cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();
	
	// get propeties of the rgb & depths.
	const uint32_t num_views  = input_rgbs.size(0);
	// the height & width of rgb, depth images.
	const uint32_t height_rgb = input_rgbs.size(2);
	const uint32_t width_rgb  = input_rgbs.size(3);
	// 
	const uint32_t height_d = input_depths.size(2);
	const uint32_t width_d  = input_depths.size(3);

	torch::Tensor dis_bgs_rgbs = torch::zeros_like( input_rgbs );
	torch::Tensor dis_ds       = torch::zeros_like( input_depths );

	// torch::Tensor remap_rgbs = torch::empty({num_views, height_rgb, width_rgb, 2}, bgs.options());
	// torch::Tensor remap_ds   = torch::empty({num_views, height_d, width_d, 2}, bgs.options());
	
	int block_size, grid_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, build_from_remap_kernel<6>, 0, 0);
    grid_size = (num_views*height_rgb*width_rgb + block_size - 1) / block_size;

	// remap kernels for depths and colors. approx. < 1ms
	build_from_remap_kernel<6> <<< grid_size, block_size, 0, curr_stream >>> (
		input_rgbs.contiguous().data_ptr<float>(),
		// inputs params.
		DIS_C.contiguous().data_ptr<float>(),
		K_colors.contiguous().data_ptr<float>(),
		// remap_rgbs.contiguous().data_ptr<float>(),
		dis_bgs_rgbs.contiguous().data_ptr<float>(),
		num_views, height_rgb, width_rgb
	);
	
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, build_from_remap_kernel<1>, 0, 0);
	grid_size = (num_views*height_d*width_d + block_size - 1) / block_size;

	build_from_remap_kernel<1> <<< grid_size, block_size, 0, curr_stream >>> (
		input_depths.contiguous().data_ptr<float>(),
		// inputs params.
		DIS_D.contiguous().data_ptr<float>(),
		K_depths.contiguous().data_ptr<float>(),
		// remap_ds.contiguous().data_ptr<float>(),
		dis_ds.contiguous().data_ptr<float>(),
		num_views, height_d, width_d
	);

	CUDA_CHECK_ERRORS();

	return {dis_bgs_rgbs, dis_ds};
}