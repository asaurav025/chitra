-- SQL queries to clear all existing data from the Chitra photo database
-- WARNING: This will delete ALL data from all tables!

-- Disable foreign key checks temporarily for faster deletion
PRAGMA foreign_keys = OFF;

-- Delete all data from tables (order matters due to foreign keys)
-- Delete child tables first, then parent tables

-- Clear face-related data
DELETE FROM face_thumbs;
DELETE FROM faces;

-- Clear photo-related data
DELETE FROM embeddings;
DELETE FROM tags;
DELETE FROM clusters;
DELETE FROM photos;

-- Clear person data (should be empty after faces are deleted, but clearing anyway)
DELETE FROM persons;

-- Re-enable foreign key checks
PRAGMA foreign_keys = ON;

-- Reset auto-increment counters (optional, but useful for clean state)
DELETE FROM sqlite_sequence WHERE name IN ('photos', 'embeddings', 'tags', 'clusters', 'persons', 'faces', 'face_thumbs');

-- Verify tables are empty (uncomment to check)
-- SELECT 'photos' as table_name, COUNT(*) as count FROM photos
-- UNION ALL SELECT 'embeddings', COUNT(*) FROM embeddings
-- UNION ALL SELECT 'tags', COUNT(*) FROM tags
-- UNION ALL SELECT 'clusters', COUNT(*) FROM clusters
-- UNION ALL SELECT 'persons', COUNT(*) FROM persons
-- UNION ALL SELECT 'faces', COUNT(*) FROM faces
-- UNION ALL SELECT 'face_thumbs', COUNT(*) FROM face_thumbs;

