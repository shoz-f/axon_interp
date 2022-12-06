defmodule AxonInterp do
  @moduledoc """
  The thin wrapper for Axon inference.
  It is a Deep Learning inference framework that can be used in the same way as my *Interp.
  """

  @basic_usage """
  ## Basic Usage
  You get the trained axon model or onnx model and save it in a directory that your application can read.
  "your-app/priv" may be good choice.

  ```
  $ cp your-trained-model.axon ./priv
  ```

  Next, you will create a module that interfaces with the deep learning model.
  The module will need pre-processing and post-processing in addition to inference
  processing, as in the example following. AxonInterp provides inference processing
  only.

  You put `use AxonInterp` at the beginning of your module, specify the model path
  as an optional argument and describe model's inputs/outputs specification. In the
  inference, you will put data input to the model (`AxonInterp.set_input_tensor/3`),
  inference execution (`AxonInterp.invoke/1`), and inference result retrieval
  (`AxonInterp.get_output_tensor/2`).

  ```elixr:your_model.ex
  defmodule YourApp.YourModel do
    use AxonInterp,
      model: "priv/your-trained-model.onnx",
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
        |> Nx.from_binary({:f, 32})
        |> Nx.reshape({size-x, size-y, :auto})

      * your-postprocessing *
      ...
    end
  end
  ```
  """

  @timeout 300000

  @framework "axon"

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
        {_, predict_fn} = Axon.build(model, mode: :inference)

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

      require Nx

      def handle_call({:invoke, inputs}, _from, %{model: model, params: params, itempl: template}=state) do
        n = Enum.count(template)

        inputs =
          if n == 1 do
            inputs[0]
          else
            Enum.map(0..n-1, &inputs[&1]) |> List.to_tuple()
          end

        results = case model.(params, inputs) do
          single when Nx.is_tensor(single) ->
            %{0 => single}
          multiple when is_tuple(multiple) ->
            Tuple.to_list(multiple)
            |> Enum.with_index()
            |> Enum.into(%{}, fn {result, index} -> {index, result} end)
        end
        
        {:reply, {:ok, results}, state}
      end

      def handle_call({:itempl, index}, _from, %{itempl: template}=state) do
        {:reply, {:ok, Enum.at(template, index)}, state}
      end

      def handle_call({:otempl, index}, _from, %{otempl: template}=state) do
        {:reply, {:ok, Enum.at(template, index)}, state}
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
  Ensure that the back-end framework is as expected.
  """
  def framework?(name) do
    unless String.downcase(name) == String.downcase(@framework),
      do: raise "Error: backend NN framework is \"#{@framework}\", not \"#{name}\"."
  end

  @doc """
  Ensure that the model matches the back-end framework.
  
  ## Parameters
    * model - path of model file
    * url - download site
  """
  def validate_model(nil, _), do: raise ArgumentError, "need a model file #{inspect(@model_suffix)}."
  def validate_model(model, url) do
    validate_extname!(model)

    abs_path = Path.expand(model)
    unless File.exists?(abs_path) do
      IO.puts("#{model}:")
      {:ok, _} = AxonInterp.URL.download(url, Path.dirname(abs_path), Path.basename(abs_path))
    end
    model
  end

  defp validate_extname!(model) do
    actual_ext = Path.extname(model)
    unless actual_ext in @model_suffix,
      do: raise ArgumentError, "#{@framework} expects the model file #{inspect(@model_suffix)} not \"#{actual_ext}\"."

    actual_ext
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
  Stop the axon interpreter.

  ## Parameters

    * mod - modules' names
  """
  def stop(mod) do
    GenServer.stop(mod)
  end

  @doc """
  Put a flat binary to the input tensor on the interpreter.

  ## Parameters

    * sessin - session record
    * index - index of input tensor in the model
    * bin   - input data - flat binary, cf. serialized tensor
    * opts
      * [:binary] - data is binary
  """
  def set_input_tensor(%AxonInterp{module: mod, inputs: inputs}=session, index, data, opts \\ []) do
    data = cond do
      :binary in opts ->
        with {:ok, {dtype, shape}} <- GenServer.call(mod, {:itempl, index}, @timeout) do
          Nx.from_binary(data, dtype) |> Nx.reshape(shape)
        end

      true -> data
      end

    %AxonInterp{session | inputs: Map.put(inputs, index, data)}
  end

  @doc """
  Get the flat binary from the output tensor on the interpreter.

  ## Parameters

    * session - session record
    * index - index of output tensor in the model
    * opts:
      * [:binary] - convert the output to binary
  """
  def get_output_tensor(%AxonInterp{outputs: outputs}, index, opts \\ []) do
    output = outputs[index]

    cond do
      :binary in opts -> Nx.to_binary(output)
      true -> output
    end
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
  def invoke(%AxonInterp{module: mod, inputs: inputs}=session) do
    case GenServer.call(mod, {:invoke, inputs}, @timeout) do
      {:ok, results} -> %AxonInterp{session | outputs: results}
      any -> any
    end
  end
end
