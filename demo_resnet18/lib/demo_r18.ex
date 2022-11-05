defmodule DemoR18 do
  @width  224
  @height 224

  use AxonInterp,
    model: "~/nn_model/resnet18.onnx",
    url: "https://github.com/shoz-f/axon_interp/releases/download/0.0.1/resnet18.onnx",
    inputs: [f32: {1, 3, @height, @width}],
    outputs: [f32: {1, 1000}]

  @imagenet1000 (for item <- File.stream!("./imagenet1000.label") do
                   String.trim_trailing(item)
                 end)
                |> Enum.with_index(&{&2, &1})
                |> Enum.into(%{})

#  @imagenet1000 AxonInterp.Util.download("https://github.com/shoz-f/axon_interp/releases/download/0.0.1/imagenet1000.label",
#                    fn x -> String.split("\n")
#                      |> Enum.with_index(&{&2, &1})
#                      |> Enum.into(%{})
#                    end
#                  )

  def apply(img, top \\ 1) do
    # preprocess
    input0 = CImg.builder(img)
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:gauss, {{123.7, 58.4}, {116.3, 57.1}, {103.5, 57.4}}}, :nchw])
                        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

    # prediction
    outputs = session()
      |> AxonInterp.set_input_tensor(0, input0)
      |> AxonInterp.invoke()
      |> AxonInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({1000})

    # postprocess
    exp = Nx.exp(outputs)

    Nx.divide(exp, Nx.sum(exp))     # softmax
    |> Nx.argsort(direction: :desc)
    |> Nx.slice([0], [top])
    |> Nx.to_flat_list()
    |> Enum.map(&@imagenet1000[&1])
  end

  def run() do
    unless File.exists?("lion.jpg"),
      do: AxonInterp.Util.download("https://github.com/shoz-f/nn-interp/releases/download/0.0.1/lion.jpg")

    CImg.load("lion.jpg")
    |> __MODULE__.apply(3)
  end
end
