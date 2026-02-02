
import unittest
import duckdb
import pandas as pd
import numpy as np
from core.modules.statistical_testing import get_feature_matrix, perform_clustering, generate_clustering_interpretation

class TestClustering(unittest.TestCase):
    def setUp(self):
        # Create in-memory DB
        self.con = duckdb.connect(':memory:')
        self.con.execute("CREATE TABLE corpus (token TEXT, filename TEXT, pos TEXT, lemma TEXT, _token_low TEXT)")
        
        # Insert dummy data
        # Group 1 (Fruit): File A, File B
        # Group 2 (Tech): File C
        
        data = []
        # File A: apple(2), orange(1), banana(1)
        data.extend([('apple', 'fileA'), ('apple', 'fileA'), ('orange', 'fileA'), ('banana', 'fileA')])
        # File B: orange(1), banana(1), strawberry(1)
        data.extend([('orange', 'fileB'), ('banana', 'fileB'), ('strawberry', 'fileB')])
        # File C: computer(2), mouse(1)
        data.extend([('computer', 'fileC'), ('computer', 'fileC'), ('mouse', 'fileC')])
        
        for token, fname in data:
            self.con.execute("INSERT INTO corpus VALUES (?, ?, 'NN', ?, ?)", (token, fname, token, token.lower()))
            
        # We need a filepath for the functions, but they take a path string to connect. 
        # Since our functions expect a path to CONNECT to, we can't easily injection an existing connection unless we patch.
        # However, DuckDB connects to ':memory:' if path is ':memory:'. But each connection is isolated?
        # No, ':memory:' is isolated.
        # Solution: Create a temporary file db.
        self.db_path = "test_corpus.db"
        self.disk_con = duckdb.connect(self.db_path)
        self.disk_con.execute("CREATE TABLE corpus (token TEXT, filename TEXT, pos TEXT, lemma TEXT, _token_low TEXT)")
        for token, fname in data:
            self.disk_con.execute("INSERT INTO corpus VALUES (?, ?, 'NN', ?, ?)", (token, fname, token, token.lower()))
        self.disk_con.close()

    def tearDown(self):
        import os
        try:
            os.remove(self.db_path)
        except:
            pass

    def test_feature_matrix_extraction(self):
        # Test get_feature_matrix
        matrix, top_words = get_feature_matrix(self.db_path, group_by='filename', top_n_features=5)
        
        print("Top words:", top_words)
        print("Matrix:\n", matrix)
        
        self.assertTrue('apple' in top_words, "apple should be a top word")
        self.assertTrue('computer' in top_words, "computer should be a top word")
        
        # Check counts
        # File A should have 2 apples
        self.assertEqual(matrix.loc['fileA', 'apple'], 2)
        # File C should have 0 apples
        self.assertEqual(matrix.loc['fileC', 'apple'], 0)
        
    def test_clustering_execution(self):
        matrix, _ = get_feature_matrix(self.db_path, group_by='filename', top_n_features=10)
        
        # Test Euclidean
        res = perform_clustering(matrix, distance_metric='euclidean', method='ward')
        self.assertTrue('linkage' in res)
        self.assertTrue('distance_matrix' in res)
        
        # Check linkage shape (n-1, 4)
        Z = res['linkage']
        self.assertEqual(Z.shape, (2, 4)) # 3 items -> 2 merges
        
        # Check first merge. Should likely be File A and File B (Fruits)
        # Indices: 0=fileA, 1=fileB, 2=fileC (alphabetical usually, or as returned)
        labels = res['labels']
        idx_a = labels.index('fileA')
        idx_b = labels.index('fileB')
        
        # The first row of Z should combine these two indices
        fused1 = int(Z[0][0])
        fused2 = int(Z[0][1])
        
        joined = {labels[fused1], labels[fused2]}
        print("First merge joined:", joined)
        
        # Ideally fileA and fileB are closer than C
        self.assertTrue('fileA' in joined and 'fileB' in joined, "Fruits should cluster together first")
        
    def test_interpretation(self):
        matrix, _ = get_feature_matrix(self.db_path, group_by='filename', top_n_features=10)
        res = perform_clustering(matrix)
        
        text = generate_clustering_interpretation(res['norm_matrix'], res['linkage'], res['labels'])
        print("Interpretation:\n", text)
        
        self.assertTrue("similar" in text.lower())
        self.assertTrue("file" in text.lower())

if __name__ == '__main__':
    unittest.main()
