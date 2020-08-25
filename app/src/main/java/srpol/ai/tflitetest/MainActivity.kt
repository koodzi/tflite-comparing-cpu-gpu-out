package srpol.ai.tflitetest

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.channels.FileChannel
import kotlin.concurrent.thread
import kotlin.math.abs

typealias Tensor4 = Array<Array<Array<FloatArray>>>

fun newTensor4(d1: Int, d2: Int, d3: Int, d4: Int, value: (Int) -> Float = {0f}): Tensor4 {
    return Array(d1) { Array(d2) { Array(d3) { FloatArray(d4) { value(it) } } } }
}

data class TestModel(
    val modelName: String,
    val modelInput: List<Int>,
    val modelOutput: List<Int>
)

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val models = listOf(
            TestModel("model_01.tflite", listOf(1, 9, 6, 45), listOf(1, 18, 6, 32)),
            TestModel("model_02.tflite", listOf(1, 9, 6, 45), listOf(1, 21, 8, 32)),

            TestModel("model_03.tflite", listOf(1, 10, 6, 45), listOf(1, 20, 6, 32)),
            TestModel("model_04.tflite", listOf(1, 10, 6, 45), listOf(1, 23, 8, 32))
        )
        runTestForModels(models)
    }

    fun runTestForModels(models: Iterable<TestModel>) {

        // create generators for each run
        val timeToRunEachModel = 3
        val timesToRun: List<(Int) -> Float> = (1..timeToRunEachModel).map { { _: Int -> Math.random().toFloat() }}

        // combine models and generators together for easier use
        val modelsValuesToTest = models.map { model -> Array(timesToRun.size) { model }.zip(timesToRun) }.flatten()

        // start - running sequential but in threads
        runTest(modelsValuesToTest.iterator())
    }

    fun runTest(runs: Iterator<Pair<TestModel, (Int) -> Float>>) {

        // run until we got models
        if (!runs.hasNext()) {
            return
        }

        thread {

            val (model, num_generator) = runs.next()

            // generate input tensor
            val (i1, i2, i3, i4) = model.modelInput
            val input = newTensor4(i1, i2, i3, i4, num_generator)

            val (o1, o2, o3, o4) = model.modelOutput

            val out1 = newTensor4(o1, o2, o3, o4)
            getOutput(getModel(model.modelName, true), input, out1)

            val out2 = newTensor4(o1, o2, o3, o4)
            getOutput(getModel(model.modelName, false), input, out2)

            runOnUiThread {
                findViewById<TextView>(R.id.result).text =  "${findViewById<TextView>(R.id.result).text} (${model.modelName}): ${diff(out1, out2)}\n"

                // next run
                runTest(runs)
            }
        }
    }

    fun getModel(model: String, useGpu: Boolean = true): Interpreter {
        var tfliteOptions = Interpreter.Options()
        tfliteOptions.setNumThreads(4)
        tfliteOptions.setUseNNAPI(false)
        tfliteOptions.setAllowFp16PrecisionForFp32(false)

        if (useGpu) {
            val gpuDelegate = GpuDelegate(
                // increase numeric stability
                GpuDelegate.Options().setPrecisionLossAllowed(false)
            )
            tfliteOptions.addDelegate(gpuDelegate)
        }

        val asset = assets.openFd(model)
        FileInputStream(asset.fileDescriptor).use {
            val modelMappedByteBuffer =  it.channel.map(
                FileChannel.MapMode.READ_ONLY,
                asset.startOffset,
                asset.declaredLength
            )
            return Interpreter(modelMappedByteBuffer, tfliteOptions).apply {
                this.resetVariableTensors()
            }
        }
    }

    fun getOutput(model: Interpreter, in1: Tensor4, out1: Tensor4) {
        model.run(in1, out1)
        model.close()
    }

    fun diff(x: Tensor4, y: Tensor4) : String {
        var sum = 0.0f
        var count = 0
        var nans1 = 0
        var nans2 = 0
        var max = -1f
        var min = 1f
        for (i in x.indices) {
            for (j in x[i].indices) {
                for (k in x[i][j].indices) {
                    for (l in x[i][j][k].indices) {

                        if (x[i][j][k][l].isNaN())
                            nans1 += 1

                        if (y[i][j][k][l].isNaN())
                            nans2 += 1

                        if (x[i][j][k][l].isNaN() || y[i][j][k][l].isNaN()) {
                            continue
                        } else {
                            val v = abs(x[i][j][k][l] - y[i][j][k][l])
                            if (max < v) {
                                max = v
                            }
                            sum += v
                            count += 1
                        }
                    }
                }
            }
        }

        return "mean diff: ${String.format("%8.3g", sum / count)} max: ${String.format("%8.3g", max)}"
    }
}