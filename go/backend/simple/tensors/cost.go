package tensors

import (
	"math"
	"neural/backend"
)

type CostFunction func(dimension int, values []float64, truths []float64, deltas []float64) float64
type CostDerivative func(dimension int, pred []float64, truths []float64, deltas []float64)

type CostActivator struct {
	Function   CostFunction
	Derivative CostDerivative
}

func MSE(dimension int, predictions []float64, truths []float64, costs []float64) float64 {
	t := float64(0)
	l := len(predictions) / dimension
	lf := float64(l)
	for i := 0; i < dimension; i++ {
		for j := 0; j < l; j++ {
			pos := (j * dimension) + i
			v0 := truths[pos]
			v1 := predictions[pos]
			costs[i] += 0.5 * math.Pow(v1-v0, 2)
		}
		costs[i] = costs[i] / lf
		t += costs[i]
	}
	return t / float64(len(costs))

}

func MSE_D(dimension int, predictions []float64, truths []float64, deltas []float64) {
	l := len(predictions) / dimension
	lf := float64(l)
	for i := 0; i < dimension; i++ {
		for j := 0; j < l; j++ {
			pos := (j * dimension) + i
			v0 := truths[pos]
			v1 := predictions[pos]
			deltas[i] += v1 - v0 // since we chose to apply 1/2 to the cost
		}
		deltas[i] = deltas[i] / lf
	}
}

var MSEActivator = CostActivator{
	Function:   MSE,
	Derivative: MSE_D,
}

var costActivatorMap = map[int]CostActivator{backend.MSECostActivatorId: MSEActivator}
