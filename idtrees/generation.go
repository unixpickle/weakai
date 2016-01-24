package idtrees

// GenerateTree returns the root TreeNode for an identification
// tree that classifies the given data set.
// If no such tree exists, it will return nil.
func GenerateTree(d DataSet) *TreeNode {
	return generateNode(d, map[Field]bool{})
}

func generateNode(d DataSet, usedFields map[Field]bool) *TreeNode {
	classes := d.classes()
	if len(classes) == 0 {
		return &TreeNode{}
	} else if len(classes) == 1 {
		return &TreeNode{LeafValue: classes[0]}
	}

	field := bestField(d, usedFields)
	if field == nil {
		return nil
	}

	res := &TreeNode{BranchField: field, Branches: map[Value]*TreeNode{}}
	usedFields[field] = true
	for _, value := range field.Values() {
		subset := d.filter(field, value)
		subnode := generateNode(subset, usedFields)
		if subnode == nil {
			return nil
		}
		res.Branches[value] = subnode
	}
	usedFields[field] = false

	return res
}

func bestField(d DataSet, usedFields map[Field]bool) Field {
	var leastDisorder float64
	var bestField Field
	for _, field := range d.allFields() {
		if usedFields[field] {
			continue
		}
		var disorder float64
		for _, value := range field.Values() {
			subset := d.filter(field, value)
			fractionOfWhole := float64(len(subset)) / float64(len(d))
			disorder += fractionOfWhole * subset.disorder()
		}
		if disorder < leastDisorder || bestField == nil {
			leastDisorder = disorder
			bestField = field
		}
	}
	return bestField
}
