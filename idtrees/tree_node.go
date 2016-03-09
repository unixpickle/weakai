package idtrees

import (
	"bytes"
	"strings"
)

// A TreeNode is one node in an identification tree.
//
// If a TreeNode is completely narrowed down to one or zero classes,
// or if no more distinctions could be made, then its BranchField
// will be nil.
//
// If a TreeNode could not be narrowed down further but was
// not all one class, then LeafValue is the majority class and
// Indeterminate is true.
//
// If a TreeNode is narrowed down to zero classes, i.e. it is
// impossible to reach, then all its fields will be nil.
//
// If a TreeNode is not narrowed down to one or fewer classes,
// then its BranchField will be non-nil, and its Branches will
// indicate children TreeNodes for each possible Value for the
// BranchField.
type TreeNode struct {
	LeafValue     Value
	Indeterminate bool

	BranchField Field
	Branches    map[Value]*TreeNode
}

// String returns a human-readable representation of the tree,
// using indentation to signify depth in the tree.
func (t *TreeNode) String() string {
	if t.Leaf() {
		if t.LeafValue != nil {
			if t.Indeterminate {
				return t.LeafValue.String() + " (Indeterminate)"
			} else {
				return t.LeafValue.String()
			}
		} else {
			return "(Unreachable)"
		}
	}

	var buf bytes.Buffer

	buf.WriteString(t.BranchField.String())
	buf.WriteRune('\n')

	isFirst := true
	for value, node := range t.Branches {
		if !isFirst {
			buf.WriteRune('\n')
		}
		isFirst = false
		var subBuf bytes.Buffer
		subBuf.WriteRune(' ')
		subBuf.WriteString(value.String())
		subBuf.WriteString(" -> ")
		subBuf.WriteString(node.String())
		buf.WriteString(strings.Replace(subBuf.String(), "\n", "\n  ", -1))
	}

	return buf.String()
}

// Leaf returns true if the TreeNode has no further branches.
// It returns true for TreeNodes with a LeafValue, and for
// TreeNodes that are entirely unreachable.
func (t *TreeNode) Leaf() bool {
	return t.BranchField == nil
}
