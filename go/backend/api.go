package backend

const (
	MSECostActivatorId = 1
)

const (
	IdentityActivatorId = 1
	ReLUActivatorId     = 2
)

type TensorsApi interface {
	CreateTensor(shape ...int) int
	BuildWeights(dx int, dy int, seed int64) int
	Copy(handle1 int, handle2 int)
	Forward(inputValuesHandle int, weightsHandle int, valuesHandle int, deltasHandle int, outputValuesHandle int, activatorId int)
	Cost(costActivatorId int, outputsHandle int, truthsHandle int, costs []float64) float64
	CostDerivative(costActivatorId int, outputsHandle int, truthsHandle int, deltasHandle int)
	ComputeDeltas(l1DeltasHandle int, l0WeightsHandle int, l0ValuesHandle int, activatorId int, l0DeltasHandle int)
	UpdateWeights(sourceValuesHandle int, weightsHandle int, deltasHandle int, alpha float64)
	PrintMatrix(tensorHandle int)
	LoadInputValues(tensorHandle int, inputs [][]float64)
}

type EmbeddingsApi interface {
	BuildEmbeddings(sequenceLen int, dimension int) int
	BuildEncodings(sequenceLen int, dimension int, num int) int
	CreateVocabulary() int
	TokenizePrompt(handle int, p string) []int
	EmbeddingsLookup(handle int, batches [][]int) int
	AddEncodings(batchHandle int, encodingsHandle int) int
	Dropout(batchHandle int, p float64, seed int64)
}
