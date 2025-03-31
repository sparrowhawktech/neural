package tensors

import "neural/backend"

func Identity(in []float64, out []float64) {
	copy(out, in)
}

func Identity_D(in float64) float64 {
	return 1
}

func ReLU(in []float64, out []float64) {
	for i, f := range in {
		if f < 0 {
			out[i] = 0
		} else {
			out[i] = f
		}
	}
}

func ReLU_D(v float64) float64 {
	if v < 0 {
		return 0
	} else {
		return 1
	}
}

type ActivationCallback func([]float64, []float64)
type DerivativeCallback func(float64) float64

type Activator struct {
	Function   ActivationCallback
	Derivative DerivativeCallback
}

var IdentityActivator = Activator{
	Function:   Identity,
	Derivative: Identity_D,
}

var ReLUActivator = Activator{
	Function:   ReLU,
	Derivative: ReLU_D,
}

var activatorMap = map[int]Activator{backend.IdentityActivatorId: IdentityActivator, backend.ReLUActivatorId: ReLUActivator}
