#include <stdio.h>
#include <vector>
#include "cvt_color.h"
#include "net.h"

struct Rect
{
    float x;
    float y;
    float width;
    float height;
};

struct Object
{
    struct Rect rect;
	int label;
	float prob;
};

static int detect_yolov2_tiny(LP_IMAGE_DATA_I pImageBmp, std::vector<Object>& objects)
{
	ncnn::Net yolov2;

	// 加载模型文件
	yolov2.load_param("yolov2_tiny_3.param");

	// 模型加载时间
	//timeval s1, s2;
	//double s;
	//gettimeofday(&s1, NULL);

	yolov2.load_model("yolov2_tiny_3.bin");

	//gettimeofday(&s2, NULL);
	//s = (s2.tv_sec - s1.tv_sec) * 1000.0;
	//s += (s2.tv_usec - s1.tv_usec) / 1000.0;
	//printf("model load time: %f ms\n", s);

	const int target_size = 416;

	int img_w = pImageBmp->mWidth;
	int img_h = pImageBmp->mHeight;
	// printf("test image size: %d %d\n", img_w, img_h);

	// 如果输入图像为3通道，则ncnn的输入在这里需要将cv::imread读到的BGR转为RGB，使用ncnn::Mat::PIXEL_BGR2RGB
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(pImageBmp->mpData, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, target_size, target_size);

	// the Caffe-YOLOv2-Windows style
	// X' = X * scale - mean
	const float mean_vals[3] = { 0.5f, 0.5f, 0.5f };
	const float norm_vals[3] = { 0.007843f, 0.007843f, 0.007843f };
	//in.substract_mean_normalize(0, norm_vals);
	//in.substract_mean_normalize(mean_vals, 0);
	in.substract_mean_normalize(mean_vals, norm_vals);

	ncnn::Extractor ex = yolov2.create_extractor();
	//ex.set_num_threads(4);

	ex.input("data", in);

	ncnn::Mat out;

	// 检测时间
	//timeval t1, t2;
	//double t;
	//gettimeofday(&t1, NULL);

	ex.extract("detection_out", out);

	//gettimeofday(&t2, NULL);
	//t = (t2.tv_sec - t1.tv_sec) * 1000.0;
	//t += (t2.tv_usec - t1.tv_usec) / 1000.0;
	//printf("detection time: %f ms\n", t);

	printf("%d %d %d\n", out.w, out.h, out.c);
	objects.clear();
	for (int i = 0; i < out.h; i++)
	{
		const float* values = out.row(i);

		Object object;
		object.label = (int)(values[0]);
		object.prob = values[1];
		object.rect.x = values[2] * img_w;
		object.rect.y = values[3] * img_h;
		object.rect.width = values[4] * img_w - object.rect.x;
		object.rect.height = values[5] * img_h - object.rect.y;

		objects.push_back(object);
	}

    static const char* class_names[] = { "hold", "stop", "shutter" };
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "class = %d,  prob = %.2f%%, rect = (%.0f, %.0f) %.0f x %.0f\n",
            obj.label, obj.prob * 100.f, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
    }

	if (objects.size() == 0)
		printf("no object\n");
	return 0;
}

int main(int argc, char** argv)
{
    char *bmpPath = "stop.bmp";
    BMP_PIC_T BMPImage;
    LoadBMP((char *)bmpPath, &BMPImage);
    assert(&BMPImage.stImage);
    LP_IMAGE_DATA_I pImageBmp = &BMPImage.stImage;

	std::vector<Object> objects;
	detect_yolov2_tiny(pImageBmp, objects);

    ReleaseBMP(&BMPImage);
    printf("Press Enter to exit\n");
    getchar();
	return 0;
}
