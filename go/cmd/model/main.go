package main

import (
	"fmt"
	"math"
	"neural/backend"
	"neural/backend/simple/tensors"
	"neural/model"
)

func main() {

	dimension := 3

	backendApi := tensors.NewApi()

	inputLayer := model.NewLayer("input", backendApi, dimension, 13, backend.IdentityActivatorId)
	l1 := model.NewLayer("l1", backendApi, dimension*3, 14, backend.IdentityActivatorId)
	outputLayer := model.NewLayer("output", backendApi, dimension, 13, backend.IdentityActivatorId)

	datasetLen := 10
	inputs := make([][]float64, datasetLen)
	truths := make([][]float64, datasetLen)
	for i := 0; i < datasetLen; i++ {
		inputs[i] = make([]float64, dimension)
		truths[i] = make([]float64, dimension)
		offset := i * dimension
		for j := 0; j < dimension; j++ {
			v := float64(offset + j + 1)
			inputs[i][j] = v
			truths[i][j] = v / 10
		}
	}

	inputLayer.LoadValues(inputs)
	l1.Connect(inputLayer)
	outputLayer.Connect(l1)

	truthsHandle := backendApi.CreateTensor(dimension, datasetLen)
	backendApi.LoadInputValues(truthsHandle, truths)

	lr := 0.001
	network := model.NewNetwork(outputLayer, truthsHandle)
	sgd := model.NewSGD(network, backendApi, backend.MSECostActivatorId, lr)
	println("WEIGHTS:")
	fmt.Printf("%v\n", backendApi.Data(*outputLayer.WeightsHandle))
	backendApi.PrintMatrix(*outputLayer.WeightsHandle)
	sgd.Forward()
	println("VALUES:")
	backendApi.PrintMatrix(*outputLayer.OutputValuesHandle)
	fmt.Printf("Cost: %.8f\n", network.Cost)
	lastCost := math.MaxFloat64
	inertia := 5
	failCount := 0
	for n := 0; n < 10000; n++ {
		println("------------------")
		println("WEIGHTS BEFORE:")
		backendApi.PrintMatrix(*outputLayer.WeightsHandle)
		sgd.Backward()
		println("WEIGHTS AFTER:")
		backendApi.PrintMatrix(*outputLayer.WeightsHandle)
		sgd.Forward()
		println("VALUES:")
		backendApi.PrintMatrix(*outputLayer.OutputValuesHandle)
		fmt.Printf("Cost: %.64f\n", network.Cost)
		if math.Abs(lastCost-network.Cost) <= lr/1000 {
			failCount++
		} else {
			failCount = 0
		}
		if failCount >= inertia {
			break
		}
		if network.Cost == 0.00000000 {
			break
		}
		if math.IsNaN(network.Cost) {
			break
		}
		lastCost = network.Cost
	}

}
