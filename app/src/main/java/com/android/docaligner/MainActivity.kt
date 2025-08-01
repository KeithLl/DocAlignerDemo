package com.android.docaligner

import com.pic.aligner.PicAligner
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.pic.consts.Constants
import com.pic.onnx.OnnxUtils
import org.opencv.android.OpenCVLoader
import java.io.File
import kotlin.math.log

class MainActivity : AppCompatActivity() {
    private val TYPE_ONNX_MODEL = "onnx"

    private var mTvDemoBtn: TextView? = null

    //    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        mTvDemoBtn = findViewById(R.id.tv_demo_btn)

        Log.e(Constants.LOG_TAG, "File.pathSeparator : ${File.separator}")
        mTvDemoBtn?.setOnClickListener {
            for (i in 10..11) {
                val fileName = "$i.png"
                val originPath = OnnxUtils.getParentDir() + File.separator + fileName
                val savedPath = OnnxUtils.getParentDir() + File.separator + "saved_" + fileName
                PicAligner.getInstance(baseContext)
                    .straightenImage(originPath, savedPath, object : PicAligner.Callback {
                        override fun onSuccess(resultPaths: String) {
                            Log.e(Constants.LOG_TAG, resultPaths)
                        }

                        override fun onError(message: String) {
                            Log.e(Constants.LOG_TAG, message)
                        }

                        override fun onProgress(step: String, progress: Float) {
                            Log.e(Constants.LOG_TAG, step + progress)
                        }

                    })
            }
        }
        requestPermission()
    }


    override fun onDestroy() {
        // 释放资源
        PicAligner.release()
        super.onDestroy()
    }

    private fun requestPermission() {
        //请求存储权限
        if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.WRITE_EXTERNAL_STORAGE), 1)
        } else {
            decideInitByType()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        if (requestCode == 1) {
            if (grantResults.isNotEmpty() && grantResults[0] == android.content.pm.PackageManager.PERMISSION_GRANTED) {
                decideInitByType()
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    private fun decideInitByType() {
        realDecide(TYPE_ONNX_MODEL)
    }

    private fun realDecide(type: String) {
        when (type) {
            TYPE_ONNX_MODEL -> initOnnxModel()
            else -> initOthers()
        }
    }

    private fun initOthers() {

    }

    private fun initOnnxModel() {
        // 初始化OpenCV
        Log.e(Constants.LOG_TAG, "initOnnxModel")
        OpenCVLoader.initLocal()
        OnnxUtils.initParentDir(context = this)
    }
}
