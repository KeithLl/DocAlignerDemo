package com.android.docaligner.onnx

import ai.onnxruntime.OnnxTensor
import android.util.Log
import com.android.docaligner.Constants.LOG_TAG
import com.google.gson.Gson
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

object PostProcess {
    fun postProcess(
        outputTensor: OnnxTensor,
        imgsSize: Pair<Int, Int>,
        heatmapThreshold: Float = 0.3f
    ): Array<Array<Double>> {

        Log.e(LOG_TAG, "====== postProcess start =======")

        // 1. 获取输出数据
        val outputData = outputTensor.floatBuffer.array()
        val outputShape = outputTensor.info.shape
        Log.e(LOG_TAG, "outputShape: ${Gson().toJson(outputShape)}")
//        Log.e(LOG_TAG, "outputData:  ${Gson().toJson(outputData)}")
        Log.e(LOG_TAG, "outputData size : ${outputData.size}")

        // 2. 解析输出数据
        val rows = outputShape[2].toInt()
        val cols = outputShape[3].toInt()
        val numPoints = outputShape[1].toInt()
        Log.e(LOG_TAG, "H: $rows, W: $cols, numPoints: $numPoints")

        val polygon = mutableListOf<List<Double>>()

        for (ii in 0 until numPoints) {
            val offset = ii * rows * cols
            val predArray = outputData.sliceArray(offset until offset + rows * cols)
            val predMat = Mat(rows, cols, CvType.CV_32FC1)
            predMat.put(0, 0, predArray)

            // 调整大小到原始图像尺寸
            val resizedPredMat = Mat()
            Imgproc.resize(
                predMat, resizedPredMat, Size(imgsSize.first.toDouble(), imgsSize.second.toDouble())
            )
            // 转换
            val saveMapTemp6 = Mat()
            (resizedPredMat).convertTo(saveMapTemp6, CvType.CV_8UC1)

            // 二值化
            var binaryPredMat = Mat()
            Core.compare(
                resizedPredMat,
                Scalar(heatmapThreshold.toDouble()),
                binaryPredMat,
                Core.CMP_GT
            )
            OnnxUtils.printAndSaveMat("stepp4$ii", binaryPredMat)

            // 获取最大面积多边形的中心点
            val point = getPointWithMaxArea(binaryPredMat)
            if (point.size == 2 && ii < 4) {
                polygon.add(point)
            }
        }

        // 将MutableList<DoubleArray>转换为Array<Array<Double>>
        return polygon.map { it.toTypedArray() }.toTypedArray()
    }

    private fun getPointWithMaxArea(mask: Mat): List<Double> {
        // 查找轮廓
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
            mask,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        if (contours.isNotEmpty()) {
            // 找到面积最大的轮廓
            var maxArea = -1.0
            var maxContourIndex = -1
            for (i in contours.indices) {
                val area = Imgproc.contourArea(contours[i])
                if (area > maxArea) {
                    maxArea = area
                    maxContourIndex = i
                }
            }

            // 计算最大面积轮廓的中心点
            val moments = Imgproc.moments(contours[maxContourIndex])
            val cx = moments.m10 / moments.m00
            val cy = moments.m01 / moments.m00
            return listOf(cx, cy)
        }

        return emptyList()
    }
}
