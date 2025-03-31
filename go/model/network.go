package model

type Network struct {
	OutputLayer  *DenseLayer
	TruthsHandle *int
	Cost         float64
	Costs        []float64
}

func NewNetwork(outputLayer *DenseLayer, truthsHandle int) *Network {
	return &Network{
		OutputLayer:  outputLayer,
		TruthsHandle: &truthsHandle,
		Cost:         0,
		Costs:        make([]float64, *outputLayer.Dimension),
	}
}
