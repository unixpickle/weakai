package idtrees

import "runtime"

const fieldCountForConcurrency = 100

// GenerateTree returns the root TreeNode for an identification
// tree that classifies the given data set.
// If no such tree exists, it will return nil.
func GenerateTree(d *DataSet) *TreeNode {
	return generateNode(d, map[Field]bool{})
}

func generateNode(d *DataSet, usedFields map[Field]bool) *TreeNode {
	classes := d.classes()
	if len(classes) == 0 {
		return &TreeNode{}
	} else if len(classes) == 1 {
		return &TreeNode{LeafValue: classes[0]}
	}

	field, fieldIdx := bestField(d, usedFields)
	if field == nil {
		return nil
	}

	res := &TreeNode{BranchField: field, Branches: map[Value]*TreeNode{}}
	usedFields[field] = true
	for _, value := range field.Values() {
		subset := d.filter(fieldIdx, value)
		subnode := generateNode(subset, usedFields)
		if subnode == nil {
			return nil
		}
		res.Branches[value] = subnode
	}
	usedFields[field] = false

	return res
}

func bestField(d *DataSet, usedFields map[Field]bool) (Field, int) {
	maxProcs := runtime.GOMAXPROCS(0)
	if len(d.Fields) < fieldCountForConcurrency || maxProcs == 1 {
		field, index, _ := bestFieldInList(d, d.Fields, 0, usedFields)
		return field, index
	}
	results := make(chan scoredField, maxProcs)
	fieldsPerRoutine := len(d.Fields) / maxProcs
	for i := 0; i < maxProcs; i++ {
		var fields []Field
		var startIndex int
		if i+1 < maxProcs {
			startIndex = i * fieldsPerRoutine
			fields = d.Fields[startIndex : (i+1)*fieldsPerRoutine]
		} else {
			startIndex = (maxProcs - 1) * fieldsPerRoutine
			fields = d.Fields[startIndex:]
		}
		go func() {
			field, index, disorder := bestFieldInList(d, fields, startIndex, usedFields)
			results <- scoredField{field, index, disorder}
		}()
	}
	var bestResult *scoredField
	for i := 0; i < maxProcs; i++ {
		res := <-results
		if bestResult == nil || bestResult.field == nil ||
			(res.disorder < bestResult.disorder && res.field != nil) {
			bestResult = &res
		}
	}
	return bestResult.field, bestResult.index
}

func bestFieldInList(d *DataSet, fields []Field, startIndex int,
	usedFields map[Field]bool) (Field, int, float64) {
	var leastDisorder float64
	var bestField Field
	var bestFieldIdx int
	for idx, field := range fields {
		if usedFields[field] {
			continue
		}
		var disorder float64
		for _, value := range field.Values() {
			partDisorder, count := d.statsForFilter(idx+startIndex, value)
			fractionOfWhole := float64(count) / float64(len(d.Entries))
			disorder += fractionOfWhole * partDisorder
		}
		if disorder < leastDisorder || bestField == nil {
			leastDisorder = disorder
			bestField = field
			bestFieldIdx = idx + startIndex
		}
	}
	return bestField, bestFieldIdx, leastDisorder
}

type scoredField struct {
	field    Field
	index    int
	disorder float64
}
