"""
Extract and analyze DiD methodological papers
"""
import os
from pathlib import Path
import PyPDF2
import json
import re

def extract_pdf_metadata(pdf_path):
    """Extract basic metadata and first page from PDF"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata = {
                'filename': pdf_path.name,
                'num_pages': len(reader.pages),
                'title': '',
                'authors': '',
                'abstract': '',
                'first_page_text': ''
            }
            
            # Get PDF metadata
            if reader.metadata:
                metadata['title'] = reader.metadata.get('/Title', '')
                metadata['authors'] = reader.metadata.get('/Author', '')
            
            # Extract first few pages text
            text = ''
            for i in range(min(3, len(reader.pages))):
                text += reader.pages[i].extract_text()
            
            metadata['first_page_text'] = text[:2000]  # First 2000 chars
            
            # Try to extract title from text if not in metadata
            if not metadata['title']:
                lines = text.split('\n')
                for line in lines[:10]:
                    if len(line) > 20 and len(line) < 200:
                        metadata['title'] = line.strip()
                        break
            
            return metadata
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def categorize_papers(papers_dir):
    """Categorize papers by analyzing filenames and content"""
    papers_dir = Path(papers_dir)
    papers = []
    
    for pdf_file in papers_dir.glob('*.pdf'):
        print(f"Processing: {pdf_file.name}")
        metadata = extract_pdf_metadata(pdf_file)
        if metadata:
            # Categorize based on filename patterns
            filename = pdf_file.name.lower()
            if 'synthetic' in filename or 'sdid' in filename:
                metadata['category'] = 'Synthetic DiD'
            elif 'two-way' in filename or 'twfe' in filename:
                metadata['category'] = 'Two-Way Fixed Effects'
            elif 'staggered' in filename or 'multiple' in filename:
                metadata['category'] = 'Staggered Adoption'
            elif 'honest' in filename or 'parallel' in filename:
                metadata['category'] = 'Parallel Trends'
            else:
                metadata['category'] = 'General DiD Methods'
            
            papers.append(metadata)
    
    return papers

def main():
    papers_dir = '/home/simon/githubRepos/DrSnow/didMethodPapers/Methodological papers on DiD'
    output_file = '/home/simon/githubRepos/DrSnow/did_papers_catalog.json'
    
    print("Extracting metadata from DiD papers...")
    papers = categorize_papers(papers_dir)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(papers, f, indent=2)
    
    print(f"\nProcessed {len(papers)} papers")
    print(f"Results saved to: {output_file}")
    
    # Print summary by category
    categories = {}
    for paper in papers:
        cat = paper['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(paper['filename'])
    
    print("\n=== Papers by Category ===")
    for cat, files in categories.items():
        print(f"\n{cat} ({len(files)} papers):")
        for f in files:
            print(f"  - {f}")

if __name__ == '__main__':
    main()
