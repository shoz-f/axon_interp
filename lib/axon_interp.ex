defmodule AxonInterp do
  @timeout 300000

  @framework "Axon"

  # the suffix expected for the model
  suffix = %{
    "axon" => [".axon", ".onnx"]
  }
  @model_suffix suffix[String.downcase(@framework)]
  
  # session record
  defstruct module: nil, inputs: %{}, outputs: %{}

  defmacro __using__(opts) do
    quote generated: true, location: :keep do
      use GenServer

      def start_link(opts) do
        GenServer.start_link(__MODULE__, opts, name: __MODULE__)
      end

      def init(opts) do
        opts = Keyword.merge(unquote(opts), opts)
        nn_model   = AxonInterp.validate_model(Keyword.get(opts, :model), Keyword.get(opts, :url))
        nn_inputs  = Keyword.get(opts, :inputs, [])
        nn_outputs = Keyword.get(opts, :outputs, [])

        {model, params} = case Path.extname(nn_model) do
          ".axon" -> File.read!(nn_model) |> Axon.deserialize()
          ".onnx" -> AxonOnnx.import(nn_model)
        end
        {_, predict_fn} = Axon.build(model, [])

        {:ok, %{model: predict_fn, params: params, path: nn_model, itempl: nn_inputs, otempl: nn_outputs}}
      end

      def session() do
        %AxonInterp{module: __MODULE__}
      end

      def handle_call({:info}, _from, state) do
        info = %{
          "model"   => state.path,
          "inputs"  => state.itempl,
          "outputs" => state.otempl,
        }
        {:reply, {:ok, info}, state}
      end
      
      def handle_call({:invoke, inputs}, _from, %{model: model, params: params, itempl: template}=state) do
        inputs = Enum.with_index(template)
          |> Enum.map(fn {{dtype, shape}, index} -> Nx.from_binary(inputs[index], dtype) |> Nx.reshape(shape) end)

        input0 = Enum.at(inputs, 0)
        result = model.(params, input0) |> Nx.to_binary()
        {:reply, {:ok, result}, state}
      end

      def terminate(_reason, state) do
        :ok
      end
    end
  end


  @doc """
  Get name of backend NN framework.
  """
  def framework() do
    @framework
  end

  @doc """
  Get the propaty of the model.

  ## Parameters
    * mod - modules' names
  """
  def info(mod) do
    case GenServer.call(mod, {:info}, @timeout) do
      {:ok, result} ->  {:ok, Map.put(result, "framework", @framework)}
      any -> any
    end
  end

  @doc """
  Stop the interpreter.

  ## Parameters
    * mod - modules' names
  """
  def stop(mod) do
    GenServer.stop(mod)
  end

  @doc """
  Put a binary to the input tensor on the interpreter.

  ## Parameters
    * sessin - session record
    * index - index of input tensor in the model
    * bin   - input data - flat binary, cf. serialized tensor
  """
  def set_input_tensor(%AxonInterp{inputs: inputs}=session, index, bin) when is_binary(bin) do
    %AxonInterp{session | inputs: Map.put(inputs, index, bin)}
  end

  @doc """
  Get the binary from the output tensor on the interpreter.

  ## Parameters
    * session - session record
    * index - index of output tensor in the model
  """
  def get_output_tensor(%AxonInterp{outputs: outputs}, index) do
    outputs[index]
  end

  @doc """
  Execute the inference session. In session mode, data input/execution of
  inference/output of results to the DL model is done all at once.

  ## Parameters
    * session - session

  ## Examples.
    ```elixir
      output_bin0 = session()
        |> AxonInterp.set_input_tensor(0, input_bin0)
        |> AxonInterp.invoke()
        |> AxonInterp.get_output_tensor(0)
    ```
  """
  def invoke(%AxonInterp{module: mod, inputs: inputs, outputs: outputs}=session) do
    case GenServer.call(mod, {:invoke, inputs}, @timeout) do
      {:ok, result} -> %AxonInterp{session | outputs: Map.put(outputs, 0, result)}
      any -> any
    end
  end

  @doc """
  Ensure that the model matches the back-end framework.
  
  ## Parameters
    * path - path of model file
    * url - download site
  """
  def validate_model(nil, _), do: raise ArgumentError, "need a model file #{inspect(@model_suffix)}."
  def validate_model(path, url) do
    path = Path.expand(path)

    actual_ext = validate_extname!(@model_suffix, path)
    unless File.exists?(path) do
      validate_extname!([actual_ext], url)
      AxonInterp.Util.download(url, Path.dirname(path), Path.basename(path))
    end
    path
  end

  defp validate_extname!(exts, path) do
    actual_ext = Path.extname(path)
    unless actual_ext in exts,
      do: raise ArgumentError, "expects the model file #{inspect(exts)} not \"#{actual_ext}\"."

    actual_ext
  end
end
