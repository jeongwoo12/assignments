#include "imgproc.h"
#include <queue>

namespace IPCVL {
	namespace IMG_PROC {
		void calcHist(cv::InputArray src, int* histogram) {
			cv::Mat inputMat = src.getMat();

			for (int y = 0; y < inputMat.rows; y++) {
				for (int x = 0; x < inputMat.cols; x++) {
					// Todo : histogram을 쌓습니다. 

					/** your code here! **/

					// hint 1 : for loop 를 이용해서 cv::Mat 순회 시 (1채널의 경우) 
					// inputMat.at<uchar>(y, x)와 같이 데이터에 접근할 수 있습니다. 
					histogram[inputMat.at<uchar>(y, x)]++;
				}
			}
		}
		void backprojectHistogram(cv::InputArray src_hsv, cv::InputArray face_hsv, cv::OutputArray dst) {
			cv::Mat srcMat = src_hsv.getMat();
			cv::Mat faceMat = face_hsv.getMat();
			dst.create(srcMat.size(), CV_8UC1);
			cv::Mat outputProb = dst.getMat();
			outputProb.setTo(cv::Scalar(0.));

			double model_hist[64][64] = { { 0., } };
			double input_hist[64][64] = { { 0., } };

			// Todo : hs 2차원 히스토그램을 계산하는 함수를 작성합니다. 
			calcHist_hs(srcMat, input_hist);
			calcHist_hs(faceMat, model_hist);

			std::vector<cv::Mat> channels;
			split(srcMat, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			for (int y = 0; y < srcMat.rows; y++) {
				for (int x = 0; x < srcMat.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 
					/** your code here! **/
					// hint 1 : UTIL::quantize()를 이용해서 srtMat의 값을 양자화합니다. 
					// hint 2 : UTIL::h_r() 함수를 이용해서 outputPorb 값을 계산합니다.
					int h = UTIL::quantize(mat_h.at<uchar>(y, x));
					int s = UTIL::quantize(mat_s.at<uchar>(y, x));

					outputProb.at<uchar>(y, x) = UTIL::h_r(model_hist, input_hist, h, s) * 255;
				}
			}
		}

		void thresh_binary(cv::InputArray src, cv::OutputArray dst, const int & threshold)
		{
			cv::Mat srcMat = src.getMat();
			dst.create(srcMat.size(), CV_8UC1);
			cv::Mat dstMat = dst.getMat();

			for (int y = 0; y < src.rows(); y++) {
				for (int x = 0; x < src.cols(); x++) {
					if (srcMat.at<uchar>(y, x) >= threshold)
						dstMat.at<uchar>(y, x) = 255;
					else
						dstMat.at<uchar>(y, x) = 0;
				}
			}
		}

		void thresh_otsu(cv::InputArray src, cv::OutputArray dst)
		{
			cv::Mat srcMat = src.getMat();
			double hist[256] = { 0., };
			double w[256] = { 0., };
			double u0[256] = { 0., };
			double u1[256] = { 0., };
			double v[256] = { 0., };

			double u = 0.;
			int L = 256;
			double max_v = 0.;
			int T = 0;

			//히스토그램 정규화
			UTIL::calcNormedHist(srcMat, hist);

			for (int i = 0; i < L; i++) {
				u += i*hist[i];
			}

			w[0] = hist[0];

			for (int t = 1; t < L; t++) {
				w[t] = w[t - 1] + hist[t];
				if (w[t] != 0.0) {
					u0[t] = ((w[t - 1] * u0[t - 1]) + (t*hist[t])) / w[t];
				}
				if (1 - w[t] != 0) {
					u1[t] = (u - (w[t] * u0[t])) / (1 - w[t]);
				}
				v[t] = w[t] * (1 - w[t])*((u0[t] - u1[t])*(u0[t] - u1[t]));
			}
			
			for (int t = 0; t < L; t++) {
				if (max_v < v[t]) {
					max_v = v[t];
					T = t;
				}
			}
			dst.create(srcMat.size(), CV_8UC1);
			cv::Mat dstMat = dst.getMat();

			for (int y = 0; y < src.rows(); y++) {
				for (int x = 0; x < src.cols(); x++) {
					if (srcMat.at<uchar>(y, x) >= T)
						dstMat.at<uchar>(y, x) = 255;
					else
						dstMat.at<uchar>(y, x) = 0;
				}
			}
		}

		void flood_fill4(cv::Mat & l, const int & j, const int & i, const int & label)
		{
			if (l.at<int>(j, i) == -1) {
				l.at<int>(j, i) = label;

				flood_fill4(l, j, i + 1, label);
				flood_fill4(l, j - 1, i, label);
				flood_fill4(l, j, i - 1, label);
				flood_fill4(l, j + 1, i, label);
			}
		}

		void flood_fill8(cv::Mat & l, const int & j, const int & i, const int & label)
		{
			if (l.at<int>(j, i) == -1) {
				l.at<int>(j, i) = label;

				flood_fill8(l, j + 1, i + 1, label);
				flood_fill8(l, j - 1, i - 1, label);
				flood_fill8(l, j - 1, i + 1, label);
				flood_fill8(l, j + 1, i - 1, label);

				flood_fill8(l, j, i + 1, label);
				flood_fill8(l, j - 1, i, label);
				flood_fill8(l, j, i - 1, label);
				flood_fill8(l, j + 1, i, label);
			}
		}

		void efficient_flood_fill4(cv::Mat & l, const int & j, const int & i, const int & label)
		{
			std::queue<cv::Point> q;
			cv::Point p;
			p.y = j;
			p.x = i;

			q.push(p);

			while (!q.empty()){
				cv::Point tmp = q.front();
				q.pop();
				if (l.at<int>(tmp.y, tmp.x) == -1) {
					int left = tmp.x;
					int right = tmp.x;

					while (l.at<int>(tmp.y, left - 1) == -1) {
						left--;
					}
					while (l.at<int>(tmp.y, right + 1) == -1) {
						right++;
					}

					for (int c = left; c <= right; c++) {

						l.at<int>(tmp.y, c) = label;
						if (l.at<int>(tmp.y - 1, c) == -1 && (c == left || l.at<int>(tmp.y - 1, c - 1) != -1)) {
							cv::Point p1;
							p1.y = tmp.y - 1;
							p1.x = c;
							q.push(p1);
						}

						if (l.at<int>(tmp.y + 1, c) == -1 && (c == left || l.at<int>(tmp.y + 1, c - 1) != -1)) {
							cv::Point p2;
							p2.y = tmp.y + 1;
							p2.x = c;
							q.push(p2);
						}	
					}
				}
			}
		}

		void flood_fill(cv::InputArray src, cv::OutputArray dst, const UTIL::CONNECTIVITIES & direction)
		{
			cv::Mat b = src.getMat(); 
			dst.create(b.size(), CV_32SC1);
			cv::Mat l = dst.getMat();

			int label = 1;

			for (int y = 0; y < l.rows; y++) {
				for (int x = 0; x < l.cols; x++) {
					if ((int)b.at<uchar>(y, x) == 0) {
						l.at<int>(y, x) = 0;
					}
					if ((int)b.at<uchar>(y, x) != 0) {
						l.at<int>(y, x) = -1;
					}
					if (y == 0 || y == l.rows - 1 || x == 0 || x == l.cols - 1) {
						l.at<int>(y, x) = 0;
					}
				}
			}

			if (direction == UTIL::CONNECTIVITIES::NAIVE_FOURWAY) {
				for (int j = 1; j < l.rows - 1; j++) {
					for (int i = 1; i < l.cols - 1; i++) {
						if (l.at<int>(j, i) == -1) {
							flood_fill4(l, j, i, label);
							label++;
						}
					}
				}
			}

			if (direction == UTIL::CONNECTIVITIES::NAIVE_EIGHT_WAY) {
				for (int j = 1; j < l.rows - 1; j++) {
					for (int i = 1; i < l.cols - 1; i++) {
						if (l.at<int>(j, i) == -1) {
							flood_fill8(l, j, i, label);
							label++;
						}
					}
				}
			}

			if (direction == UTIL::CONNECTIVITIES::EFFICIENT_FOURWAY) {
				for (int j = 1; j < l.rows - 1; j++) {
					for (int i = 1; i < l.cols - 1; i++) {
						if (l.at<int>(j, i) == -1) {
							efficient_flood_fill4(l, j, i, label);
							label++;
						}
					}
				}
			}
		}

		void calcHist_hs(cv::InputArray src_hsv, double histogram[][64]) {
			cv::Mat hsv = src_hsv.getMat();
			std::vector<cv::Mat> channels;
			split(hsv, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			// 2차원 히스토그램을 쌓습니다. 
			for (int y = 0; y < hsv.rows; y++) {
				for (int x = 0; x < hsv.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 
					/** your code here! **/
					int h = UTIL::quantize(mat_h.at<uchar>(y, x));
					int s = UTIL::quantize(mat_s.at<uchar>(y, x));
					// hint 1 : 양자화 시 UTIL::quantize() 함수를 이용해서 mat_h, mat_s의 값을 양자화시킵니다. 
					histogram[h][s]++;
				}
			}

			// 히스토그램을 (hsv.rows * hsv.cols)으로 정규화합니다. 
			for (int j = 0; j < 64; j++) {
				for (int i = 0; i < 64; i++) {
					// Todo : histogram에 있는 값들을 순회하며 (hsv.rows * hsv.cols)으로 정규화합니다. 
					/** your code here! **/
					histogram[j][i] = histogram[j][i] / (hsv.rows * hsv.cols);
				}
			}
		}
	}  // namespace IMG_PROC<<<<<<< HEAD
/*
	namespace UTIL {
		int quantize(int a) {
			int L = 256;
			int q = 64;
			return floor((a * q) / L);
		}

		double h_r(double model_hist[][64], double input_hist[][64], int j, int i) {
			double h_m = model_hist[j][i];
			double h_i = input_hist[j][i];
			double val = 0.0;

			if (h_i == 0.0) return 1.0;
			else return (double)std::min(h_m / h_i, 1.0);
		}

		void GetHistogramImage(int* histogram, cv::OutputArray dst, int hist_w, int hist_h) {
			dst.create(cv::Size(hist_w, hist_h), CV_8UC1);
			cv::Mat histImage = dst.getMat();
			histImage.setTo(cv::Scalar(255, 255, 255));

			int bin_w = cvRound((double)hist_w / 256);

			// find the maximum intensity element from histogram
			int max = histogram[0];

			for (int i = 1; i < 256; i++)
				if (max < histogram[i]) max = histogram[i];

			// normalize the histogram between 0 and histImage.rows
			for (int i = 0; i < 255; i++)
				histogram[i] = ((double)histogram[i] / max) * histImage.rows;

			// draw the intensity line for histogram
			for (int i = 0; i < 255; i++)
				cv::line(histImage, cv::Point(bin_w*(i), hist_h), cv::Point(bin_w*(i), hist_h - histogram[i]), cv::Scalar(0, 0, 0), 1, 8, 0);
		}
	}  // namespace UTIL
	*/
}