package com.android.docaligner.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.android.docaligner.Constants.LOG_TAG
import com.google.gson.Gson
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer

/**
 * Created by KeithLee on 2025/2/8.
 * Introduction:
 */
class OnnxModel {
    private lateinit var environment: OrtEnvironment
    private lateinit var session: OrtSession
    private var context: Context
    private val imageSizeInfer = Pair(256, 256)
    private lateinit var imageSizeOri: Pair<Int, Int>
    private var startTime = 0L

//    private val onnxName = "fastvit_sa24_h_e_bifpn_256_fp32.onnx"
//    private val onnxName = "fastvit_t8_h_e_bifpn_256_fp32.onnx"
        private val onnxName = "lcnet100_h_e_bifpn_256_fp32.onnx"

    constructor(context: Context) {
        this.context = context
        try {
            environment = OrtEnvironment.getEnvironment()
            val modelStream = context.assets.open(onnxName)
            val modelBytes = ByteArray(modelStream.available())
            modelStream.read(modelBytes)
            session = environment.createSession(modelBytes)
            Log.e(LOG_TAG, "OnnxModel init success" + session)
        } catch (e: Exception) {
            Log.e(LOG_TAG, "OnnxModel init error")
            e.printStackTrace()
        }
    }

    fun handleOnnx() {

        Log.e(LOG_TAG, "====== runOnnx start =======")
        startTime = System.currentTimeMillis()

        //1. 读取图片 TODO: dynamic pic path
        val step1Mat = readPic("4.png")
//        val step1Mat = readPic("pic4.jpg")
        if (step1Mat.empty()) {
            Log.e(LOG_TAG, "readPic error")
            return
        }
        OnnxUtils.printAndSaveMat("step1Mat", step1Mat)
        // init imageSizeOri
        imageSizeOri = Pair(step1Mat.width(), step1Mat.height())

        //2. RBG2BGR
        val step2Mat = OnnxUtils.rgb2bgr(step1Mat)
        OnnxUtils.printAndSaveMat("step2Mat", step2Mat)

        //3. preprocess
        val imgFloatArray = PreProcess.preprocess(step2Mat, imageSizeInfer)
        Log.e(LOG_TAG, "inputData: ${Gson().toJson(imgFloatArray)}")

        //4. runInference
        val outputTensor = runInference(imgFloatArray)

        //5. postProcess
        val result = PostProcess.postProcess(outputTensor, imageSizeOri)

        //6. print result TODO: handle result data
        Log.e(LOG_TAG, "result: ${Gson().toJson(result)}")
        //[[38.06658386562574,1679.0861485687249],[1351.0566787726273,1390.4360147559266],[1539.218103926406,2199.6418899535256],[193.42584005953404,2510.6460491919015]]
        if (result.isNotEmpty()) {
            // 多边形转正,然后保存为新的图片
            rotatePolygon(result, step1Mat)
            // 绘制多边形, 并保存图片
            drawAndSavePic(result, step1Mat)
        }
        Log.e(LOG_TAG, "====== runOnnx end =======")
        val endTime = System.currentTimeMillis()
        Log.e(LOG_TAG, "runOnnx cost time: ${endTime - startTime} ms")
    }

    private fun rotatePolygon(result: Array<Array<Double>>, step1Mat: Mat) {
        // 从result中获取点坐标
        val points = mutableListOf<org.opencv.core.Point>()
        for (point in result) {
            points.add(org.opencv.core.Point(point[0].toDouble(), point[1].toDouble()))
        }

        // 检查点数量是否足够（至少需要4个点形成多边形）
        if (points.size < 4) {
            Log.e(LOG_TAG, "多边形点数量不足，无法进行透视变换")
            return
        }

        // 对多边形顶点进行排序（按顺时针方向）
        val sortedPoints = sortPolygonPoints(points)

        // 计算原始多边形的实际宽度和高度（保持宽高比）
        val width = calculateDistance(sortedPoints[0], sortedPoints[1]) // 上边长作为宽度
        val height = calculateDistance(sortedPoints[1], sortedPoints[2]) // 右边长作为高度

        // 检查宽高有效性
        if (width <= 0 || height <= 0) {
            Log.e(LOG_TAG, "无效的多边形尺寸，无法进行透视变换")
            return
        }

        // 定义目标矩形的四个顶点（保持原始宽高比）
        val dstPoints = listOf(
            org.opencv.core.Point(0.0, 0.0),
            org.opencv.core.Point(width, 0.0),
            org.opencv.core.Point(width, height),
            org.opencv.core.Point(0.0, height)
        )

        // 转换为OpenCV需要的点格式
        val srcMat = MatOfPoint2f(*sortedPoints.toTypedArray())
        val dstMat = MatOfPoint2f(*dstPoints.toTypedArray())

        // 计算透视变换矩阵
        val perspectiveMatrix = Imgproc.getPerspectiveTransform(srcMat, dstMat)

        // 应用透视变换，拉正图像（使用原始宽高比）
        val straightenedMat = Mat()
        Imgproc.warpPerspective(
            step1Mat,
            straightenedMat,
            perspectiveMatrix,
            org.opencv.core.Size(width, height)
        )

        // 保存拉正后的图像
        OnnxUtils.printAndSaveMat("straightened_image", straightenedMat)
    }

    // 辅助函数：计算两点间距离
    private fun calculateDistance(p1: org.opencv.core.Point, p2: org.opencv.core.Point): Double {
        val dx = p2.x - p1.x
        val dy = p2.y - p1.y
        return Math.sqrt(dx * dx + dy * dy)
    }

    // 辅助函数：对多边形顶点按顺时针排序
    private fun sortPolygonPoints(points: List<org.opencv.core.Point>): List<org.opencv.core.Point> {
        // 计算质心（中心点）
        val centroid = org.opencv.core.Point(
            points.map { it.x }.average(),
            points.map { it.y }.average()
        )

        // 按极角排序（顺时针方向）
        return points.sortedWith(compareBy { point ->
            Math.atan2(point.y - centroid.y, point.x - centroid.x)
        })
    }

    private fun drawAndSavePic(result: Array<Array<Double>>, step1Mat: Mat) {
        // 从result中获取点坐标,然后绘制一个多边形
        val points = mutableListOf<org.opencv.core.Point>()
        for (point in result) {
            points.add(org.opencv.core.Point(point[0].toDouble(), point[1].toDouble()))
        }
// 转换为OpenCV需要的点格式
        val matOfPoint = MatOfPoint()
        matOfPoint.fromList(points)

        // 绘制红色多边形 (OpenCV中颜色通道为BGR, 红色为(0,0,255))
        Imgproc.polylines(
            step1Mat,
            listOf(matOfPoint),
            true,  // 是否闭合多边形
            Scalar(0.0, 0.0, 255.0),  // 红色线条
            2,  // 线条宽度
            Imgproc.LINE_AA  // 抗锯齿线条
        )

        // 保存绘制后的图像
        OnnxUtils.printAndSaveMat("result_with_polygon", step1Mat)
    }


    //运行模型
    private fun runInference(imgFloatArray: FloatArray): OnnxTensor {

        // 1. 创建输入张量
        val inputName = session.inputInfo.keys.iterator().next()
        val inputTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(imgFloatArray),
            longArrayOf(1, 3, imageSizeInfer.first.toLong(), imageSizeInfer.second.toLong())
        )

        // 2. 运行模型
        val results: OrtSession.Result = session.run(mapOf(inputName to inputTensor))

        // 3. 获取输出张量
        val outputName = session.outputInfo.keys.iterator().next()
        val outputTensor = results.get(outputName).get() as OnnxTensor
        return outputTensor
    }

    // 整体第一步，读取图片
    private fun readPic(path: String): Mat {
        val imagePath = OnnxUtils.getParentDir() + "/" + path
        val image: Mat = Imgcodecs.imread(imagePath)
        return image
    }

    fun release() {
        session.close()
        environment.close()
    }
}
