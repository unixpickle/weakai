package idtrees

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
	var leastDisorder float64
	var bestField Field
	var bestFieldIdx int
	for idx, field := range d.Fields {
		if usedFields[field] {
			continue
		}
		var disorder float64
		for _, value := range field.Values() {
			partDisorder, count := d.statsForFilter(idx, value)
			fractionOfWhole := float64(count) / float64(len(d.Entries))
			disorder += fractionOfWhole * partDisorder
		}
		if disorder < leastDisorder || bestField == nil {
			leastDisorder = disorder
			bestField = field
			bestFieldIdx = idx
		}
	}
	return bestField, bestFieldIdx
}
