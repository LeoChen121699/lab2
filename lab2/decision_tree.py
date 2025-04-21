import numpy as np
import pandas as pd
from collections import Counter
import json
import os
import pickle

# Custom JSON encoder for handling NumPy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for handling NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class DecisionTreeNode:
    """Decision tree node class"""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, gain=None):
        self.feature = feature        # Split feature
        self.threshold = threshold    # Split threshold
        self.left = left              # Left subtree
        self.right = right            # Right subtree
        self.value = value            # Leaf node value
        self.gain = gain              # Information gain at this node
    
    def is_leaf(self):
        """Check if node is a leaf"""
        return self.value is not None


class DecisionTree:
    """Decision tree model implementation"""
    
    def __init__(self, max_depth=None, min_samples_split=2, min_information_gain=1e-7, max_features=None):
        """
        Parameters:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split
            min_information_gain: Minimum information gain required for split
            max_features: Maximum number of features to consider for split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.max_features = max_features
        self.root = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        """Train the decision tree model"""
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(set(y))
        
        # Check stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            return DecisionTreeNode(value=self._most_common_label(y))
        
        # Select feature subset to consider
        feature_idxs = np.arange(n_features)
        if self.max_features is not None and self.max_features < n_features:
            feature_idxs = np.random.choice(feature_idxs, self.max_features, replace=False)
        
        # Find best split
        best_feature, best_thresh, best_gain = self._best_split(X, y, feature_idxs)
        
        # Stop if information gain is too small
        if best_gain < self.min_information_gain:
            return DecisionTreeNode(value=self._most_common_label(y))
        
        # Recursively build left and right subtrees
        left_idxs = X[:, best_feature] <= best_thresh
        right_idxs = ~left_idxs
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return DecisionTreeNode(feature=best_feature, threshold=best_thresh, 
                               left=left, right=right, gain=best_gain)
    
    def _best_split(self, X, y, feature_idxs):
        """Find best feature and threshold for split"""
        best_gain = -float("inf")
        best_feature, best_thresh = None, None
        
        # Calculate parent node entropy
        parent_entropy = self._entropy(y)
        
        # For each feature
        for feature_idx in feature_idxs:
            # Get all thresholds for this feature
            thresholds = np.unique(X[:, feature_idx])
            
            # For each possible threshold
            for threshold in thresholds:
                # Split data
                left_idxs = X[:, feature_idx] <= threshold
                right_idxs = ~left_idxs
                
                # Skip if one side has no samples
                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue
                
                # Calculate child node entropies
                left_entropy = self._entropy(y[left_idxs])
                right_entropy = self._entropy(y[right_idxs])
                
                # Calculate sample weights
                n = len(y)
                n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
                
                # Calculate information gain
                info_gain = parent_entropy - (n_left / n) * left_entropy - (n_right / n) * right_entropy
                
                # Update best split
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature_idx
                    best_thresh = threshold
        
        return best_feature, best_thresh, best_gain
    
    def _entropy(self, y):
        """Calculate entropy"""
        n = len(y)
        if n == 0:
            return 0
        
        # Calculate probability of each class
        counter = Counter(y)
        probs = [count / n for count in counter.values()]
        
        # Calculate entropy: -sum(p * log2(p))
        entropy = -sum(p * np.log2(p) for p in probs)
        return entropy
    
    def _most_common_label(self, y):
        """Return most common class"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict class for samples"""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree for prediction"""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def print_tree(self, indent="  ", feature_names=None):
        """Print tree structure"""
        if feature_names is None:
            feature_names = self.feature_names
        
        def _print_node(node, depth=0):
            if node.is_leaf():
                print(f"{indent * depth}Predicted class: {node.value}")
                return
            
            feature_name = feature_names[node.feature]
            print(f"{indent * depth}If {feature_name} <= {node.threshold} (information gain = {node.gain:.4f}):")
            _print_node(node.left, depth + 1)
            print(f"{indent * depth}Else:")
            _print_node(node.right, depth + 1)
        
        _print_node(self.root)

    def save_model(self, filepath):
        """Save model to file"""
        model_data = {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_information_gain': self.min_information_gain,
            'max_features': self.max_features,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump((model_data, self.root), f)
        
        print(f"Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data, root = pickle.load(f)
        
        model = DecisionTree(
            max_depth=model_data['max_depth'],
            min_samples_split=model_data['min_samples_split'],
            min_information_gain=model_data['min_information_gain'],
            max_features=model_data['max_features']
        )
        model.feature_names = model_data['feature_names']
        model.root = root
        
        return model
    
    def export_rules(self, filepath, feature_names=None):
        """Export decision tree rules to readable format"""
        if feature_names is None:
            feature_names = self.feature_names
        
        rules = []
        
        def _extract_rules(node, rule=None, depth=0):
            if rule is None:
                rule = []
            
            if node.is_leaf():
                rules.append({
                    'conditions': rule,
                    'prediction': node.value,
                    'depth': depth
                })
                return
            
            # Left subtree rules
            left_rule = rule.copy()
            feature_name = feature_names[node.feature]
            left_rule.append(f"{feature_name} <= {node.threshold}")
            _extract_rules(node.left, left_rule, depth + 1)
            
            # Right subtree rules
            right_rule = rule.copy()
            right_rule.append(f"{feature_name} > {node.threshold}")
            _extract_rules(node.right, right_rule, depth + 1)
        
        _extract_rules(self.root)
        
        # Save rules to JSON format using custom encoder for NumPy types
        with open(filepath, 'w') as f:
            json.dump(rules, f, indent=2, cls=NumpyEncoder)
        
        print(f"Decision rules exported to: {filepath}")


class RandomForest:
    """Random Forest model implementation"""
    
    def __init__(self, n_trees=5, max_depth=None, min_samples_split=2, 
                 min_information_gain=1e-7, max_features='sqrt', bootstrap=True):
        """
        Parameters:
            n_trees: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum number of samples required to split
            min_information_gain: Minimum information gain required for split
            max_features: Maximum number of features to consider for split ('sqrt', 'log2', int or None)
            bootstrap: Whether to use bootstrap sampling
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        """Train the random forest model"""
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        
        n_samples, n_features = X.shape
        
        # Determine max_features value
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        else:
            max_features = n_features
        
        # Train each tree
        self.trees = []
        for _ in range(self.n_trees):
            # Create decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_information_gain=self.min_information_gain,
                max_features=max_features
            )
            
            # Bootstrap sampling
            if self.bootstrap:
                idxs = np.random.choice(n_samples, n_samples, replace=True)
                X_tree, y_tree = X[idxs], y[idxs]
            else:
                X_tree, y_tree = X, y
            
            # Train tree
            tree.fit(X_tree, y_tree, self.feature_names)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """Predict class for samples"""
        # Make predictions for each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Return voting results
        return np.array([
            Counter(tree_preds[:, i]).most_common(1)[0][0] 
            for i in range(X.shape[0])
        ])
    
    def save_model(self, filepath):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_information_gain': self.min_information_gain,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'feature_names': self.feature_names
        }
        
        tree_data = [{'root': tree.root} for tree in self.trees]
        
        with open(filepath, 'wb') as f:
            pickle.dump((model_data, tree_data), f)
        
        print(f"Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data, tree_data = pickle.load(f)
        
        # Create random forest
        forest = RandomForest(
            n_trees=model_data['n_trees'],
            max_depth=model_data['max_depth'],
            min_samples_split=model_data['min_samples_split'],
            min_information_gain=model_data['min_information_gain'],
            max_features=model_data['max_features'],
            bootstrap=model_data['bootstrap']
        )
        forest.feature_names = model_data['feature_names']
        
        # Rebuild each tree
        forest.trees = []
        for tree_info in tree_data:
            tree = DecisionTree(
                max_depth=model_data['max_depth'],
                min_samples_split=model_data['min_samples_split'],
                min_information_gain=model_data['min_information_gain']
            )
            tree.feature_names = model_data['feature_names']
            tree.root = tree_info['root']
            forest.trees.append(tree)
        
        return forest 