defmodule Resnet18 do
  @width  224
  @height 224

  use AxonInterp,
    model: "./model/resnet18.onnx",
    url: "https://github.com/shoz-f/axon_interp/releases/download/0.0.1/resnet18.onnx",
    inputs: [f32: {1, 3, @height, @width}],
    outputs: [f32: {1, 1000}]

  @imagenet1000 (for item <- File.stream!("./model/imagenet1000.label") do
                     String.trim_trailing(item)
                   end)
                   |> Enum.with_index(&{&2, &1})
                   |> Enum.into(%{})

  def apply(img, top \\ 1) do
    # preprocess
    input0 = CImg.builder(img)
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:gauss, {{123.7, 58.4}, {116.3, 57.1}, {103.5, 57.4}}}, :nchw])

    # prediction
    output0 = session()
      |> AxonInterp.set_input_tensor(0, input0, [:binary])
      |> AxonInterp.invoke()
      |> AxonInterp.get_output_tensor(0)
      |> Nx.squeeze()

    # postprocess
    then(Nx.exp(output0), fn exp -> Nx.divide(exp, Nx.sum(exp)) end)     # softmax
    |> Nx.argsort(direction: :desc)
    |> Nx.slice([0], [top])
    |> Nx.to_flat_list()
    |> Enum.map(&@imagenet1000[&1])
  end
end
