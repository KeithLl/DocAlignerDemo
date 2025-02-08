package com.android.docaligner

import android.annotation.SuppressLint
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.keith.android.onnx.OnnxModel
import com.keith.android.onnx.OnnxUtils
import org.opencv.android.OpenCVLoader

class MainActivity : AppCompatActivity() {
    val TYPE_ONNX_MODEL = "onnx"
    private lateinit var onnxModel: OnnxModel
    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        requestPermission()
    }


    override fun onDestroy() {
        // 释放资源
        onnxModel?.release()
        super.onDestroy()
    }

    private fun requestPermission() {
        //请求存储权限
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(android.Manifest.permission.WRITE_EXTERNAL_STORAGE), 1)
            } else {
                decideInitByType()
            }
        } else {
            decideInitByType()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
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
        try {
            OnnxUtils.initParentDir(context = this)
            OpenCVLoader.initLocal()
        } catch (e: Exception) {
            e.printStackTrace()
        }

        onnxModel = OnnxModel(this)
        onnxModel.handleOnnx()
    }
}
