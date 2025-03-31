package embeddings

import (
	"math"
	"math/rand"
	"neural/backend/simple/tensors"
	"strings"
)

type Api struct {
	tensorsApi   *tensors.Api
	vocabularies []*Vocabulary
	seed         *int64
}

func (o *Api) TokenizePrompt(vocabularyHandle int, prompt string) []int {
	vocabulary := o.vocabularies[vocabularyHandle-1]
	return vocabulary.RegisterWords(strings.Split(prompt, " ")) // debería quedar sorted  cross-bacth
}

func (o *Api) BuildEmbeddings(sequenceLen int, dimension int) int {
	tensorId := o.tensorsApi.CreateTensor(sequenceLen, dimension)
	capacity := sequenceLen * dimension
	data := o.tensorsApi.Data(tensorId)
	source := rand.NewSource(*o.seed)
	r := rand.New(source)
	di := 0
	for j := 0; j < sequenceLen; j++ {
		for k := 0; k < dimension; k++ {
			n := r.Intn(capacity)
			di++
			data[di] = 1.0 / float64(n)
		}
	}
	return tensorId
}

func (o *Api) BuildEncodings(sequenceLen int, dimension int, num int) int {
	handle := o.tensorsApi.CreateTensor(sequenceLen, dimension)
	data := o.tensorsApi.Data(handle)
	dm := float64(dimension)
	n := float64(num)
	vectorOffset := 0
	for k := 0; k < dimension; k++ {
		kf := float64(k)
		vectorOffset = k * dimension
		for i := 0; i < dimension/2; i++ {
			di := float64(i)
			theta := kf / math.Pow(n, (2*di)/dm)
			data[vectorOffset+2*i] = math.Sin(theta)
			data[vectorOffset+(2*i)+1] = math.Cos(theta)
		}
	}
	return handle
}

func (o *Api) CreateVocabulary() int {
	vocabulary := NewVocabulary()
	id := len(o.vocabularies) + 1
	o.vocabularies = append(o.vocabularies, vocabulary)
	return id
}

func (o *Api) EmbeddingsLookup(embeddingsHandle int, batches [][]int) int {
	embeddingsShape := o.tensorsApi.Shape(embeddingsHandle)
	batchCount := len(batches)
	sequenceLen := embeddingsShape[1]
	dimension := embeddingsShape[2]
	embeddingsData := o.tensorsApi.Data(embeddingsHandle)
	tensorHandle := o.tensorsApi.CreateTensor(embeddingsShape...)
	data := o.tensorsApi.Data(tensorHandle)
	di := 0
	id := 0
	embeddingOffset := 0
	for i := 0; i < batchCount; i++ {
		batch := batches[i]
		l := len(batch)
		for j := 0; j < sequenceLen; j++ {
			if j >= l {
				embeddingOffset = 1
			} else {
				id = batch[j]
				embeddingOffset = (id - 1) * dimension
			}
			for k := 0; k < dimension; k++ {
				di++
				if embeddingOffset > -1 {
					data[di] = embeddingsData[embeddingOffset+k]
				} else {
					data[di] = 0
				}
			}
		}
	}
	return tensorHandle
}

func (o *Api) AddEncodings(batchHandle int, encodingsHandle int) int {
	l := len(embeddings)
	result := make([][]float64, l)
	for k := 0; k < l; k++ {
		pev := make([]float64, dimension)
		for i := 0; i < dimension; i++ {
			pev[i] = embeddings[k][i] + vectors[k][i]
		}
		result[k] = pev
	}
	return result
	return 0
}

func (o *Api) Dropout(batchHandle int, p float64, seed int64) {
	total := len(batch) * dimension
	max := int(float64(total) * p)
	rnd := rand.New(rand.NewSource(seed))
	locations := make([]int, max)
	for i := 0; i < max; i++ {
		location := rnd.Intn(total)
		l := locations[i]
		for l == location {
			location = rnd.Intn(total)
		}
		locations[i] = location
	}

	for _, l := range locations {
		x := l / dimension
		y := l - (x * dimension)
		batch[x][y] = 0
	}
}

func NewEmbeddingsApi(tensorsApi *tensors.Api, seed int64) *Api {
	return &Api{tensorsApi: tensorsApi, vocabularies: make([]*Vocabulary, 0), seed: &seed}
}
