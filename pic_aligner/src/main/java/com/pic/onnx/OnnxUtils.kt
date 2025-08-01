package com.pic.onnx

import android.content.Context
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

/**
 * Created by KeithLee on 2025/2/8.
 * Introduction:
 */
object OnnxUtils {

    private lateinit var finalPath: String

    fun initParentDir(context: Context) {
        finalPath = context.getExternalFilesDir("knowbox")?.absolutePath ?: ""
    }

    fun getParentDir(): String {
        return finalPath
    }

    fun saveImage(savePath: String, mat: Mat): Boolean {
        return Imgcodecs.imwrite(savePath, mat)
    }

    fun readImage(imagePath: String): Mat {
        return Imgcodecs.imread(imagePath)
    }

    fun resizeMat(img: Mat, size: Pair<Int, Int>): Mat {
        val imgResized = Mat()
        Imgproc.resize(img, imgResized, Size(size.first.toDouble(), size.second.toDouble()))
        return imgResized
    }

    fun rgb2bgr(imgRgb: Mat): Mat {
        val imgBGR = Mat()
        Imgproc.cvtColor(imgRgb, imgBGR, Imgproc.COLOR_BGR2RGB)
        return imgBGR
    }
}
