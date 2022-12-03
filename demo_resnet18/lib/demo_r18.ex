defmodule DemoR18 do
  def run(top \\ 3) do
    prepare_img([
      {"lion.jpg", "https://github.com/shoz-f/nn-interp/releases/download/0.0.1/lion.jpg"}
    ])
    |> Enum.each(fn file ->
      CImg.load(file)
      |> Resnet18.apply(top)
      |> IO.inspect(label: "#{file} is ")
    end)
  end
  
  defp prepare_img(list) do
    Enum.map(list, fn {file, url} ->
      unless File.exists?(file), do: AxonInterp.URL.download(url)
      file
    end)
  end
end
