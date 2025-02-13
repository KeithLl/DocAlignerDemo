package com.android.docaligner.onnx

import android.util.Log
import com.android.docaligner.Constants.LOG_TAG
import org.opencv.core.Mat

object PreProcess {
    private fun transposeAndNormalize(img: Mat): FloatArray {

        // 获取图像的高度、宽度和通道数
        val height = img.height()
        val width = img.width()
        val channels = img.channels()

        // 创建一个用于存储处理后数据的FloatArray
        val outputSize = 1 * channels * height * width
        val outputArray = FloatArray(outputSize)

        // 遍历图像的每个像素
        for (c in 0 until channels) {
            for (h in 0 until height) {
                for (w in 0 until width) {
                    // 获取像素值并归一化
                    val pixelValue = img.get(h, w)[c].toFloat() / 255.0f
                    // 计算在outputArray中的索引
                    val index = c * height * width + h * width + w
                    outputArray[index] = pixelValue
                }
            }
        }

        return outputArray
    }

    fun preprocess(step2Mat: Mat, size: Pair<Int, Int>): FloatArray {
        Log.e(LOG_TAG, "====== preprocess start =======")

        val resizeMat = OnnxUtils.resizeMat(step2Mat, size)
        OnnxUtils.printAndSaveMat("step3Mat", resizeMat)

        val transposeMat = transposeAndNormalize(resizeMat)
        return transposeMat
    }
}
