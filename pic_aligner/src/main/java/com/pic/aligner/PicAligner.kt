package com.pic.aligner

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.annotation.SuppressLint
import com.pic.onnx.OnnxUtils
import com.pic.onnx.PostProcess
import com.pic.onnx.PreProcess
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.atan2
import kotlin.math.sqrt

class PicAligner private constructor(context: Context) {
    companion object {

        @SuppressLint("StaticFieldLeak")
        @Volatile
        private var instance: PicAligner? = null

        // 模型文件名（与OnnxModel保持一致的assets加载方式）
        private const val MODEL_NAME_FASTVIT_T8 = "fastvit_t8_h_e_bifpn_256_fp32.onnx"
        private const val MODEL_NAME_LCNET = "lcnet100_h_e_bifpn_256_fp32.onnx"

        fun getInstance(context: Context): PicAligner {
            if (instance == null) {
                synchronized(PicAligner::class.java) {
                    if (instance == null) {
                        // 检查OpenCV初始化状态（与OnnxModel.initOnnxModel()保持一致）
                        if (!OpenCVLoader.initLocal()) {
                            throw IllegalStateException("OpenCV初始化失败")
                        }
                        instance = PicAligner(context)
                    }
                }
            }
            return instance!!
        }

        fun release() {
            instance?.let {
                it.session1.close()
                it.session2.close()
                it.environment.close()
                // OrtEnvironment无需手动关闭（单例管理）
                instance = null
            }
        }
    }

    // 模型相关成员（与OnnxModel保持一致的初始化方式）
    private var environment: OrtEnvironment
    private var session1: OrtSession  // 轮廓检测模型1
    private var session2: OrtSession  // 轮廓检测模型2
    private var context = context.applicationContext
    private val imageSizeInfer = Pair(256, 256)
    private val expectSize = 4

    init {
        try {
            // 1. 初始化OrtEnvironment（与OnnxModel保持一致）
            environment = OrtEnvironment.getEnvironment()

            // 2. 加载第一个模型（轮廓检测）
            session1 = loadModel(MODEL_NAME_FASTVIT_T8)

            // 3. 加载第二个模型（轮廓检测2）
            session2 = loadModel(MODEL_NAME_LCNET)

        } catch (e: Exception) {
            throw RuntimeException("模型初始化失败", e)
        }
    }

    /**
     * 绘制轮廓
     */
    fun drawContour(
        inputImagePath: String, outputImagePath: String, callback: Callback?
    ) {

        // 1. 读取图片
        val originalMat = OnnxUtils.readImage(inputImagePath)

        // 2. 处理图片,识别顶点坐标
        val resultData = getImgPointsArray(originalMat, callback)
        if (resultData.isEmpty()) {
            callback?.onError("轮廓检测为空")
            return
        }

        // 3. 从result中获取点坐标List
        val points = getPoints(resultData)
        callback?.onProgress("获取坐标完成", 0.6f)

        // 4. 转换为OpenCV需要的点格式
        val matOfPoint = MatOfPoint()
        matOfPoint.fromList(points)
        callback?.onProgress("坐标点转换完成", 0.7f)

        // 5. 绘制红色多边形 (OpenCV中颜色通道为BGR, 红色为(0,0,255))
        Imgproc.polylines(
            originalMat, listOf(matOfPoint), true,  // 是否闭合多边形
            Scalar(0.0, 0.0, 255.0),  // 红色线条
            2,  // 线条宽度
            Imgproc.LINE_AA  // 抗锯齿线条
        )
        callback?.onProgress("绘制轮廓完成", 0.8f)

        // 6. 保存图片
        val saveResult = OnnxUtils.saveImage(outputImagePath, originalMat)
        callback?.onProgress("保存图片结果", 0.9f)
        if (saveResult) {
            callback?.onSuccess(outputImagePath)
        } else {
            callback?.onError("保存图片失败")
        }
    }

    /**
     * 图像拉正
     */
    fun straightenImage(
        inputImagePath: String, outputImagePath: String, callback: Callback?
    ) {
        // 1. 读取图片
        val originalMat = OnnxUtils.readImage(inputImagePath)
        // 2. 处理图片,识别顶点坐标
        // 使用第一个模型进行轮廓检测,如果数据为空则使用第二个模型进行检测
        val resultData = getImgPointsArray(originalMat, callback)
        if (resultData.isEmpty()) {
            callback?.onError("轮廓检测为空")
            return
        }

        // 3. 从result中获取点坐标List
        val points = getPoints(resultData)
        callback?.onProgress("获取坐标完成", 0.6f)
        // 4. 检查点数量是否足够（至少需要4个点形成多边形）
        if (points.size < 4) {
            callback?.onError("多边形点数量不足，无法进行透视变换")
            return
        }

        // 5. 对多边形顶点进行排序（按顺时针方向）
        val sortedPoints = sortPolygonPoints(points)

        // 6. 计算原始多边形的实际宽度和高度（保持宽高比）
        val width = calculateDistance(sortedPoints[0], sortedPoints[1]) // 上边长作为宽度
        val height = calculateDistance(sortedPoints[1], sortedPoints[2]) // 右边长作为高度

        // 检查宽高有效性
        if (width <= 0 || height <= 0) {
            callback?.onError("多边形尺寸无效，无法进行透视变换")
            return
        }
        callback?.onProgress("获取多边形宽高", 0.7f)

        // 7. 定义目标矩形的四个顶点（保持原始宽高比）
        val dstPoints = listOf(
            Point(0.0, 0.0),
            Point(width, 0.0),
            Point(width, height),
            Point(0.0, height)
        )

        // 转换为OpenCV需要的点格式
        val srcMat = MatOfPoint2f(*sortedPoints.toTypedArray())
        val dstMat = MatOfPoint2f(*dstPoints.toTypedArray())

        // 8. 计算透视变换矩阵
        val perspectiveMatrix = Imgproc.getPerspectiveTransform(srcMat, dstMat)

        // 应用透视变换，拉正图像（使用原始宽高比）
        val straightenedMat = Mat()
        Imgproc.warpPerspective(
            originalMat,
            straightenedMat,
            perspectiveMatrix,
            Size(width, height)
        )
        callback?.onProgress("透视变换并拉正图像", 0.8f)

        // 9. 保存拉正后的图像
        val saveResult = OnnxUtils.saveImage(outputImagePath, straightenedMat)
        callback?.onProgress("保存图片结果", 0.9f)
        if (saveResult) {
            callback?.onSuccess(outputImagePath)
        } else {
            callback?.onError("保存图片失败")
        }
    }

    // 以下为私有辅助方法（均仿照OnnxModel和OnnxUtils中的实现）
    /**
     * 模型加载通用方法（抽取OnnxModel中的模型加载逻辑）
     */
    private fun loadModel(modelName: String): OrtSession {
        val modelStream = context.assets.open(modelName)
        val modelBytes = ByteArray(modelStream.available())
        modelStream.read(modelBytes)
        return environment.createSession(modelBytes)
    }

    /**
     * 图像处理主方法（仿照OnnxModel.handleOnnx()流程）
     */

    private fun processImage(
        session: OrtSession,
        originalMat: Mat, callback: Callback?
    ): Array<Array<Double>> {
        try {
            // 1. 读取图片（复用OnnxUtils的路径处理逻辑）
            if (originalMat.empty()) {
                callback?.onError("读取图片失败")
                return emptyArray()
            }
            callback?.onProgress("图片读取完成", 0.1f)

            // 2. RBG2BGR
            val processMat = OnnxUtils.rgb2bgr(originalMat)
            callback?.onProgress("图片转换完成", 0.2f)

            // 3. 预处理（仿照OnnxModel的preprocess流程）
            val preprocessedData = preprocessImage(processMat)
            callback?.onProgress("图像预处理完成", 0.3f)

            // 4. 轮廓检测模型推理（对应OnnxModel.runInference）
            val contourResult = runModelInference(session, preprocessedData)
            callback?.onProgress("轮廓模型推理完成", 0.4f)

            // 5. 轮廓后处理（仿照OnnxModel.postProcess）
            val contours = postProcessContour(
                contourResult, Pair(originalMat.width(), originalMat.height())
            )
            callback?.onProgress("轮廓结果处理完成", 0.5f)
            return contours
        } catch (e: Exception) {
            callback?.onError("处理图片失败: ${e.message}")
            return emptyArray()
        } finally {
        }
    }


    // 获取图片顶点坐标
    private fun getImgPointsArray(
        originalMat: Mat,
        callback: Callback?,
    ): Array<Array<Double>> {
        val resultData: Array<Array<Double>>
        val resultData1 = processImage(session1, originalMat, callback)
        if (resultData1.isNotEmpty() && resultData1.size == expectSize) {
            resultData = resultData1
        } else {
            val resultData2 = processImage(session2, originalMat, callback)
            resultData = if (resultData2.isNotEmpty() && resultData2.size == expectSize) {
                resultData2
            } else {
                if (resultData1.size > resultData2.size) resultData1 else resultData2
            }
        }
        return resultData
    }

    private fun getPoints(resultData: Array<Array<Double>>): List<Point> {
        // 从result中获取点坐标
        val points = mutableListOf<Point>()
        for (point in resultData) {
            points.add(Point(point[0], point[1]))
        }

        return points
    }

    // 辅助函数：计算两点间距离
    private fun calculateDistance(p1: Point, p2: Point): Double {
        val dx = p2.x - p1.x
        val dy = p2.y - p1.y
        return sqrt(dx * dx + dy * dy)
    }

    // 辅助函数：对多边形顶点按顺时针排序
    private fun sortPolygonPoints(points: List<Point>): List<Point> {
        // 计算质心（中心点）
        val centroid = Point(
            points.map { it.x }.average(),
            points.map { it.y }.average()
        )

        // 按极角排序（顺时针方向）
        return points.sortedWith(compareBy { point ->
            atan2(point.y - centroid.y, point.x - centroid.x)
        })
    }

    /**
     * 图像预处理（仿照OnnxModel.PreProcess.preprocess）
     */
    private fun preprocessImage(srcMat: Mat): FloatArray {
        return PreProcess.preprocess(srcMat, imageSizeInfer)
    }

    /**
     * 模型推理通用方法（抽取OnnxModel.runInference逻辑）
     */
    private fun runModelInference(session: OrtSession, inputData: FloatArray): OnnxTensor {
        // 1. 创建输入张量
        val inputName = session.inputInfo.keys.iterator().next()
        val inputTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(inputData),
            longArrayOf(1, 3, imageSizeInfer.first.toLong(), imageSizeInfer.second.toLong())
        )

        // 2. 运行模型
        val results: OrtSession.Result = session.run(mapOf(inputName to inputTensor))

        // 3. 获取输出张量
        val outputName = session.outputInfo.keys.iterator().next()
        val outputTensor = results.get(outputName).get() as OnnxTensor
        return outputTensor
    }

    /**
     * 轮廓后处理（仿照OnnxModel.PostProcess.postProcess）
     */
    private fun postProcessContour(
        outputTensor: OnnxTensor, originalSize: Pair<Int, Int>
    ): Array<Array<Double>> {
        return PostProcess.postProcess(outputTensor, originalSize)
    }

    /**
     * 回调接口（扩展OnnxModel的日志输出为回调机制）
     */
    interface Callback {
        fun onSuccess(resultPaths: String)  // 返回两个结果路径
        fun onError(message: String)
        fun onProgress(step: String, progress: Float)
    }
}
