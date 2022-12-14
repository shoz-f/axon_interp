defmodule YOLOv4 do
  @moduledoc """
  Original work:
    Pytorch-YOLOv4 - https://github.com/Tianxiaomo/pytorch-YOLOv4
  """

  @width  608
  @height 608

  alias AxonInterp, as: NNInterp
  use NNInterp,
    model: "./model/yolov4_1_3_608_608_static.onnx",
    url: "https://github.com/shoz-f/axon_interp/releases/download/0.0.1/yolov4_1_3_608_608_static.onnx",
    inputs: [f32: {1,3,@width,@height}],
    outputs: [f32: {1,22743,1,4}, f32: {1,22743,80}]

  def apply(img) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:range, {0.0, 1.0}}, :nchw])

    # prediction
    outputs = session()
      |> NNInterp.set_input_tensor(0, input0, [:binary])
      |> NNInterp.invoke()

    # postprocess
    boxes  = NNInterp.get_output_tensor(outputs, 0, [:binary])
    scores = NNInterp.get_output_tensor(outputs, 1, [:binary])

    PostDNN.non_max_suppression_multi_class(
      {22743,80}, boxes, scores,
      boxrepr: :corner,
      label: "./model/coco.label"
    )
  end
end
