defmodule AxonInterp.Util do
  @doc """
  Download the file from url.
  
  ## Parameters
    * url - url of download site
    * path -
    * name -
  """
  def download(url, path \\ "./", name \\ nil) do
    IO.puts("Downloading \"#{url}\".")

    with {:ok, res} <- HTTPoison.get(url, [], follow_redirect: true),
      bin <- res.body,
      {_, <<"attachment; filename=", fname::binary>>} <- List.keyfind(res.headers, "Content-Disposition", 0),
      :ok <- File.mkdir_p(path)
    do
      Path.join(path, name||fname)
      |> save(bin)
    end
  end

  defp save(file, bin) do
    with :ok <- File.write(file, bin) do
      IO.puts("...finish.")
      {:ok, file}
    end
  end

  @doc """
  Create tensor from binary with `dtype` and `shap`.
  
  ## Parameters
    * dtype - data type of elements. {:f32, :i32, :u8}
    * shape - shape of the tensor.
    * bin   - raw data.
  
  ## Examples
    ```elixir
    > cast_tensor({:f32, {1,3,224,224}}, binary)
    ```
  """
  def create_tensor({dtype, shape}, bin) do
    Nx.from_binary(bin, dtype) |> Nx.reshape(shape)
  end
end
