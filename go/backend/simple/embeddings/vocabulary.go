package embeddings

type Vocabulary struct {
	idMap map[string]int
	maxId int
}

func (o *Vocabulary) RegisterWords(tokens []string) []int {
	l := len(tokens)
	result := make([]int, l)
	for i, t := range tokens {
		id, ok := o.idMap[t]
		if !ok {
			id = o.maxId + 1
			o.idMap[t] = id
			o.maxId = id
		}
		result[i] = id
	}
	return result
}

func NewVocabulary() *Vocabulary {
	return &Vocabulary{idMap: make(map[string]int), maxId: -1}

}
