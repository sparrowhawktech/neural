package tensors

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"neural/backend"
	"os"
	"reflect"
	"sparrowhawktech/toolkit/util"
)

type Tensor struct {
	Shape []int
	Data  []float64
}

type Api struct {
	tensorMap map[int]Tensor
	lastId    int
}

func (o *Api) CreateTensor(shape ...int) int {
	size := 1
	for _, d := range shape {
		size = size * d
	}
	data := make([]float64, size)
	id := o.lastId + 1
	o.tensorMap[id] = Tensor{
		Shape: shape,
		Data:  data,
	}
	o.lastId = id
	return id
}

func (o *Api) BuildWeights(dx int, dy int, seed int64) int {
	handle := o.CreateTensor(dx, dy)
	rnd := rand.New(rand.NewSource(seed))
	data := o.Data(handle)
	max := dx * dy
	for i := 0; i < dx; i++ {
		offset := i * dy
		for j := 0; j < dy; j++ {
			n := rnd.Intn(max)
			w := 1.0 / float64(n)
			if w == math.Inf(0) {
				w = 1 / float64(dx)
			}
			data[offset+j] = w
		}
	}
	return handle
}

func (o *Api) Shape(handle int) []int {
	return o.tensorMap[handle].Shape
}

func (o *Api) Data(id int) []float64 {
	return o.tensorMap[id].Data
}

func (o *Api) LoadInputValues(handle int, input [][]float64) {
	tensor := o.tensorMap[handle]
	data := tensor.Data
	for i, row := range input {
		offset := i * tensor.Shape[0]
		for j, v := range row {
			data[offset+j] = v
		}
	}
}

func (o *Api) Copy(handle1 int, handle2 int) {
	t1 := o.tensorMap[handle1]
	t2 := o.tensorMap[handle2]
	if !reflect.DeepEqual(t1.Shape, t2.Shape) {
		panic("Source and destination dimensions don't match")
	}
	copy(t2.Data, t1.Data)
}

func (o *Api) Forward(inputValuesHandle int, weightsHandle int, valuesHandle int, deltasHandle int, outuputValuesHandle int, activatorId int) {
	o.zero(o.Data(valuesHandle))
	weightsData := o.Data(weightsHandle)
	inputValuesTensor := o.tensorMap[inputValuesHandle]
	inputData := inputValuesTensor.Data
	valuesTensor := o.tensorMap[valuesHandle]
	valuesData := valuesTensor.Data
	deltasData := o.Data(deltasHandle)
	inputDimension := inputValuesTensor.Shape[0]
	inputCardinality := inputValuesTensor.Shape[1]
	valuesDimension := valuesTensor.Shape[0]
	for i := 0; i < valuesDimension; i++ {
		weightsOffset := i * inputDimension
		for j := 0; j < inputDimension; j++ {
			weight := weightsData[weightsOffset+j]
			for k := 0; k < inputCardinality; k++ {
				pos := (k * inputDimension) + j
				sourceValue := inputData[pos]
				value := weight * sourceValue
				if value == math.Inf(0) {
					value = 0
				}
				pos = (k * valuesDimension) + i
				valuesData[pos] += value
			}
		}
		deltasData[i] = 0
	}
	outputValuesTensor := o.tensorMap[outuputValuesHandle]
	if activatorId == backend.IdentityActivatorId {
		copy(outputValuesTensor.Data, valuesData)
	} else {
		activator := activatorMap[activatorId]
		activator.Function(valuesData, outputValuesTensor.Data)
	}
}

func (o *Api) zero(data []float64) {
	for i := 0; i < len(data); i++ {
		data[i] = 0
	}
}

func (o *Api) Cost(costActivatorId int, outputsHandle int, truthsHandle int, costs []float64) float64 {
	outputsTensor := o.tensorMap[outputsHandle]
	outputsData := outputsTensor.Data
	truthsTensor := o.tensorMap[truthsHandle]
	truthsData := truthsTensor.Data
	costActivator := costActivatorMap[costActivatorId]
	return costActivator.Function(outputsTensor.Shape[0], outputsData, truthsData, costs)

}

func (o *Api) PrintMatrix(handle int) {
	tensor := o.tensorMap[handle]
	o.doPrintMatrix(tensor, os.Stdout)
}

func (o *Api) doPrintMatrix(tensor Tensor, w io.Writer) {
	dimension := tensor.Shape[0]
	l := tensor.Shape[1] * dimension
	n := 0
	for i := 0; i < l; i++ {
		if n == dimension {
			util.WriteString("\n", w)
			n = 0
		}
		v := tensor.Data[i]
		util.WriteString(fmt.Sprintf("%*s%0.4f", n, "", v), w)
		n++
	}
	util.WriteString("\n", w)

}

func (o *Api) CostDerivative(costActivatorId int, outputsHandle int, truthsHandle int, deltasHandle int) {
	costActivator := costActivatorMap[costActivatorId]
	outputsTensor := o.tensorMap[outputsHandle]
	truthsTensor := o.tensorMap[truthsHandle]
	deltasTensor := o.tensorMap[deltasHandle]
	costActivator.Derivative(outputsTensor.Shape[0], outputsTensor.Data, truthsTensor.Data, deltasTensor.Data)
}

func (o *Api) ComputeDeltas(l1DeltasHandle int, l0WeightsHandle int, l0ValuesHandle int, activatorId int, l0DeltasHandle int) {
	l1DeltasTensor := o.tensorMap[l1DeltasHandle]
	l1DeltasData := l1DeltasTensor.Data
	l0DeltasTensor := o.tensorMap[l0DeltasHandle]
	l0DeltasData := l0DeltasTensor.Data
	weightsData := o.Data(l0WeightsHandle)
	l0ValuesTensor := o.tensorMap[l0ValuesHandle]
	dim0 := l0DeltasTensor.Shape[0]
	avgDFZs := make([]float64, dim0)
	activator := activatorMap[activatorId]
	for k := 0; k < dim0; k++ {
		avgDFZs[k] = o.computeAvgDFZ(k, l0ValuesTensor, activator.Derivative)
	}
	dim1 := l1DeltasTensor.Shape[0]

	sums := make([]float64, dim0)
	for j := 0; j < dim0; j++ {
		offset := j * dim1
		avgDfz := avgDFZs[j]
		for k := 0; k < dim1; k++ {
			d1 := l1DeltasData[k]
			weight := weightsData[offset+k]
			sums[j] -= weight * d1 * avgDfz
		}
	}
	for i := 0; i < len(sums); i++ {
		l0DeltasData[i] = sums[i] / float64(l0ValuesTensor.Shape[1])
	}
}

func (o *Api) computeAvgDFZ(nodeIndex int, tensor Tensor, derivative DerivativeCallback) float64 {
	s := float64(0)
	values := tensor.Data
	dimension := tensor.Shape[0]
	l := tensor.Shape[1]
	for k := 0; k < l; k++ {
		pos := (k * dimension) + nodeIndex
		v := values[pos]
		s += derivative(v)
	}
	return s / float64(l)
}

func (o *Api) UpdateWeights(sourceValuesHandle int, weightsHandle int, deltasHandle int, alpha float64) {
	sourceOutputValuesTensor := o.tensorMap[sourceValuesHandle]
	deltasTensor := o.tensorMap[deltasHandle]
	deltasData := deltasTensor.Data
	weightsTensor := o.tensorMap[weightsHandle]
	weightsData := weightsTensor.Data

	dim1 := deltasTensor.Shape[0]
	dim0 := sourceOutputValuesTensor.Shape[0]
	avgs := make([]float64, dim0)
	for i := 0; i < dim0; i++ {
		avgs[i] = o.computeAvgOutputValue(i, sourceOutputValuesTensor)
	}
	for i := 0; i < dim1; i++ {
		delta := deltasData[i]
		offset := i * dim0
		for j := 0; j < dim0; j++ {
			sourceAvg := avgs[j]
			pos := offset + j
			weight := weightsData[pos]
			weightsData[pos] = weight - (alpha * sourceAvg * delta)
		}
	}
}

func (o *Api) computeAvgOutputValue(nodeIndex int, tensor Tensor) float64 {
	s := float64(0)
	data := tensor.Data
	dimension := tensor.Shape[0]
	l := tensor.Shape[1]
	for i := 0; i < l; i++ {
		s += data[i*dimension+nodeIndex]
	}
	return s / float64(l)
}

func NewApi() *Api {
	return &Api{
		tensorMap: make(map[int]Tensor),
		lastId:    0,
	}
}
