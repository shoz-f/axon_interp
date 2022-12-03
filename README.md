# AxonInterp
The thin wrapper for Axon inference.
It is a Deep Learning inference framework that can be used in the same way as my *Interp.

[**ATTENSION**: AxonInterp v0.1.0 is experimental release!]

## Platform
I have confirmed it works in the following OS environment.

- WSL2/Ubuntu 20.04

## Installation
You can install it by adding `axon_interp` to the `mix.exs` dependency list as follows:

```elixir
def deps do
  [
    {:axon_interp, "~> 0.1.0"}
  ]
end
```

## Basic Usage
You get the trained Axon model or Onnx model and save it in a directory that your application can read.
"your-app/priv" may be good choice.

```
$ cp your-trained-model.axon ./priv
```

Next, you will create a module that interfaces with the deep learning model. The module will need pre-processing and
post-processing in addition to inference processing, as in the example following. AxonInterp provides inference processing only.

You put `use AxonInterp` at the beginning of your module, specify the model's path as an optional argument and describe
model's inputs/outputs specification. In the inference, you will put data input to the model (`AxonInterp.set_input_tensor/3`),
inference execution (`AxonInterp.invoke/1`), and inference result retrieval (`AxonInterp.get_output_tensor/2`).

Currentrly inputs/outputs of the medel are binary data format, because of easy to share the module with other *Interp.

```elixr:your_model.ex
defmodule YourApp.YourModel do
  use AxonInterp,
    model: "priv/your-trained-model.axon",
    inputs: [f32: {1, 3, 224, 224}],
    outputs: [f32: {1, 1000}]

  def predict(data) do
    # preprocess
    #  to convert the data to be inferred to the input format of the model.
    input_bin = convert-float32-binaries(data)

    # inference
    #  typical I/O data for Onnx models is a serialized 32-bit float tensor.
    output_bin = session()
      |> AxonInterp.set_input_tensor(0, input_bin)
      |> AxonInterp.invoke()
      |> AxonInterp.get_output_tensor(0)

    # postprocess
    #  add your post-processing here.
    #  you may need to reshape output_bin to tensor at first.
    tensor = output_bin
      |> Nx.from_binary(:f32)
      |> Nx.reshape({1000})

    * your-postprocessing *
    ...
  end
end
```

## Demo
There is Object Detection: YOLOv4 example in the demo_yolov4 directory.
This demo artistically converts a photo of a frog 'flog.jpg' in the demo directory and saves it as 'candy.jpg'.

First, you download the trained DNN model "candy-9.onnx" from the following URL and place it in the demo directory.

- [candy-9.onnx: https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx](https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx)

You can run the demo by following these steps.

```shell
$ cd demo
$ mix deps.get
$ mix run -e "DemoCandy.run"
```

You get "candy.jpg" in current directory.

Let's enjoy ;-)

## License
AxonInterp is licensed under the Apache License Version 2.0.
