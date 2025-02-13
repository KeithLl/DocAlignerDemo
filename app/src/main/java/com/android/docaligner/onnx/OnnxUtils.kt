package com.android.docaligner.onnx

import android.content.Context
import android.util.Log
import com.android.docaligner.Constants.LOG_TAG
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

    fun printAndSaveMat(name : String, mat: Mat){
        Log.e(LOG_TAG, "--------print mat--name: $name--------")
        Log.e(LOG_TAG, "img: size: ${mat.size()} , type: ${mat.type()}, nativeObj: ${mat.nativeObj}")
//        Log.e(LOG_TAG, "~ save mat ~")
        val matPath = "$finalPath/${name}.png"
        Imgcodecs.imwrite(matPath, mat)
        Log.e(LOG_TAG, "--------print and save end--------")
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
