package com.android.docaligner.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.android.docaligner.Constants.LOG_TAG
import com.google.gson.Gson
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
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

    private val onnxName = "fastvit_sa24_h_e_bifpn_256_fp32.onnx"
//    private val onnxName = "fastvit_t8_h_e_bifpn_256_fp32.onnx"
//        private val onnxName = "lcnet100_h_e_bifpn_256_fp32.onnx"

    constructor(context: Context) {
        this.context = context
        try {
            environment = OrtEnvironment.getEnvironment()
            val modelStream = context.assets.open(onnxName)
            val modelBytes = ByteArray(modelStream.available())
            modelStream.read(modelBytes)
            session = environment.createSession(modelBytes)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun handleOnnx() {

        Log.e(LOG_TAG, "====== runOnnx start =======")
        startTime = System.currentTimeMillis()

        //1. 读取图片 TODO: dynamic pic path
        val step1Mat = readPic("pic2.jpg")
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
        Log.e(LOG_TAG, "====== runOnnx end =======")
        val endTime = System.currentTimeMillis()
        Log.e(LOG_TAG, "runOnnx cost time: ${endTime - startTime} ms")
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
