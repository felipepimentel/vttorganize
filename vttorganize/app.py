import os
import logging
import argparse
import gzip
import json
import shutil
import csv
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Iterator, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
import re
from functools import lru_cache
from dataclasses import dataclass, asdict
from tqdm import tqdm
import chardet
from jinja2 import Template

import webvtt


TIME_PATTERN = re.compile(r'^\d{2}:\d{2}:\d{2}\.\d{3}$')
CACHE_FILE = Path(".vttorganize_cache.json")
MAX_CAPTION_LENGTH = 10000
BACKUP_DIR = Path("backups")
CACHE_EXPIRY_DAYS = 7
MIN_MEMORY_AVAILABLE = 500 * 1024 * 1024  # 500MB


class VTTProcessingError(Exception):
    """Erro específico para processamento de arquivos VTT."""
    pass


class OutputFormatError(Exception):
    """Erro específico para formato de saída inválido."""
    pass


@dataclass
class CaptionEntry:
    start: str
    end: str
    text: str
    file_hash: str

    def validate(self) -> None:
        if not validate_time_format(self.start):
            raise ValueError(f"Invalid start time format: {self.start}")
        if not validate_time_format(self.end):
            raise ValueError(f"Invalid end time format: {self.end}")
        if len(self.text) > MAX_CAPTION_LENGTH:
            raise ValueError(f"Caption text exceeds maximum length of {MAX_CAPTION_LENGTH} characters")
        if not self.text.strip():
            raise ValueError("Empty caption text")


@dataclass
class CacheEntry:
    entries: List[Dict[str, Any]]
    timestamp: float
    file_hash: str


class CaptionCache:
    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self.cache: Dict[str, CacheEntry] = self._load_cache()

    def _load_cache(self) -> Dict[str, CacheEntry]:
        if not self.cache_file.exists():
            return {}
        try:
            data = json.loads(self.cache_file.read_text())
            cache = {}
            for key, value in data.items():
                cache[key] = CacheEntry(
                    entries=value['entries'],
                    timestamp=value['timestamp'],
                    file_hash=value['file_hash']
                )
            return cache
        except Exception:
            return {}

    def save_cache(self):
        cache_data = {
            key: {
                'entries': value.entries,
                'timestamp': value.timestamp,
                'file_hash': value.file_hash
            }
            for key, value in self.cache.items()
        }
        self.cache_file.write_text(json.dumps(cache_data))

    def get_cached_entries(self, file_path: Path, file_hash: str) -> Optional[List[CaptionEntry]]:
        key = str(file_path)
        if key not in self.cache:
            return None

        entry = self.cache[key]
        if entry.file_hash != file_hash:
            return None

        # Verificar expiração do cache
        if datetime.fromtimestamp(entry.timestamp) + timedelta(days=CACHE_EXPIRY_DAYS) < datetime.now():
            del self.cache[key]
            return None

        return [CaptionEntry(**e) for e in entry.entries]

    def cache_entries(self, file_path: Path, entries: List[CaptionEntry], file_hash: str):
        self.cache[str(file_path)] = CacheEntry(
            entries=[asdict(entry) for entry in entries],
            timestamp=datetime.now().timestamp(),
            file_hash=file_hash
        )
        self.save_cache()


def calculate_file_hash(file_path: Path) -> str:
    import hashlib
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def setup_logging(level=logging.INFO, log_file: Optional[Path] = None):
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def detect_encoding(file_path: Path) -> str:
    with file_path.open('rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'


def process_vtt_file(filepath: Path, cache: CaptionCache) -> Iterator[Tuple[str, str, str]]:
    logging.info(f"Processing file: {filepath}")
    try:
        file_hash = calculate_file_hash(filepath)
        cached_entries = cache.get_cached_entries(filepath, file_hash)
        
        if cached_entries:
            logging.info(f"Using cached entries for {filepath}")
            for entry in cached_entries:
                yield entry.start, entry.end, entry.text
            return

        encoding = detect_encoding(filepath)
        vtt = webvtt.read(str(filepath))
        entries = []
        
        for caption in vtt:
            entry = CaptionEntry(
                start=caption.start,
                end=caption.end,
                text=caption.text.strip(),
                file_hash=file_hash
            )
            try:
                entry.validate()
                entries.append(entry)
                yield entry.start, entry.end, entry.text
            except ValueError as e:
                logging.warning(f"Skipping invalid caption in {filepath}: {str(e)}")
        
        cache.cache_entries(filepath, entries, file_hash)
    except Exception as e:
        logging.error(f"Error processing {filepath}: {str(e)}")
        raise


def merge_vtt_files(directory: Path, max_workers: Optional[int] = None) -> List[Tuple[str, str, str]]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    vtt_files = sorted(directory.glob("*.vtt"))
    
    if not vtt_files:
        raise ValueError(f"No VTT files found in {directory}")

    cache = CaptionCache()
    merged_content = []
    total_files = len(vtt_files)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_vtt_file, filepath, cache): filepath 
            for filepath in vtt_files
        }
        
        with tqdm(total=total_files, desc="Processing VTT files") as pbar:
            for future in as_completed(futures):
                filepath = futures[future]
                try:
                    for caption in future.result():
                        merged_content.append(caption)
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Failed to process {filepath}: {str(e)}")
                    raise

    merged_content.sort(key=lambda x: x[0])
    return merged_content


def validate_time_format(time_str: str) -> bool:
    return bool(TIME_PATTERN.match(time_str))


def time_to_seconds(time_str: str) -> int:
    if not validate_time_format(time_str):
        raise ValueError(f"Invalid time format: {time_str}. Expected format: HH:MM:SS.mmm")
    
    try:
        h, m, s = map(float, time_str.split(":"))
        return int(h * 3600 + m * 60 + s)
    except ValueError as e:
        raise ValueError(f"Error converting time: {time_str}") from e


def create_backup(file_path: Path) -> Path:
    """Cria um backup do arquivo antes de modificá-lo."""
    if not file_path.exists():
        return None
        
    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{file_path.stem}_{timestamp}{file_path.suffix}"
    shutil.copy2(file_path, backup_path)
    return backup_path


def find_vtt_files(directory: Path, recursive: bool = False) -> List[Path]:
    """Encontra arquivos VTT no diretório, opcionalmente de forma recursiva."""
    pattern = "**/*.vtt" if recursive else "*.vtt"
    return sorted(directory.glob(pattern))


def generate_json_content(grouped_content: List[Tuple[str, str, str]]) -> str:
    """Gera conteúdo no formato JSON."""
    json_data = []
    for start, end, text in grouped_content:
        json_data.append({
            "start_time": start,
            "end_time": end,
            "text": text
        })
    return json.dumps(json_data, indent=2, ensure_ascii=False)


def generate_csv_content(grouped_content: List[Tuple[str, str, str]]) -> str:
    output = []
    output.append(['Start Time', 'End Time', 'Text'])
    for start, end, text in grouped_content:
        output.append([start, end, text])
    
    import io
    output_buffer = io.StringIO()
    writer = csv.writer(output_buffer)
    writer.writerows(output)
    return output_buffer.getvalue()


DEFAULT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
    <style>
        body { 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            font-family: Arial, sans-serif;
        }
        .timestamp { 
            color: #666;
            font-size: 0.9em;
        }
        .section { 
            margin-bottom: 20px;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    {% for section in sections %}
    <div class="section">
        <div class="timestamp">[{{ section.start }} - {{ section.end }}]</div>
        <p>{{ section.text }}</p>
    </div>
    {% endfor %}
</body>
</html>
"""


def generate_html_content(grouped_content: List[Tuple[str, str, str]], template_path: Optional[Path] = None) -> str:
    template_str = DEFAULT_HTML_TEMPLATE
    if template_path and template_path.exists():
        template_str = template_path.read_text()

    template = Template(template_str)
    sections = [
        {'start': start, 'end': end, 'text': text}
        for start, end, text in grouped_content
    ]
    
    return template.render(
        title="Transcript",
        sections=sections
    )


def format_output(
    grouped_content: List[Tuple[str, str, str]], 
    format_type: str,
    template_path: Optional[Path] = None
) -> str:
    if not grouped_content:
        raise ValueError("No content to generate output")
    
    format_handlers = {
        "markdown": generate_markdown_content,
        "txt": generate_text_content,
        "json": generate_json_content,
        "html": lambda x: generate_html_content(x, template_path),
        "csv": generate_csv_content
    }
    
    handler = format_handlers.get(format_type)
    if not handler:
        raise OutputFormatError(f"Unsupported output format: {format_type}")
    
    return handler(grouped_content)


def generate_text_content(grouped_content: List[Tuple[str, str, str]]) -> str:
    content = "Transcript\n\n"
    for start, end, text in grouped_content:
        content += f"[{start} - {end}]\n{text}\n\n"
    return content


def generate_markdown_content(grouped_content: List[Tuple[str, str, str]]) -> str:
    content = "# Transcript\n\n"
    for start, end, text in grouped_content:
        content += f"### [{start} - {end}]\n\n{text}\n\n"
    return content


def group_captions_by_interval(
    merged_content: List[Tuple[str, str, str]], 
    interval: int = 60
) -> List[Tuple[str, str, str]]:
    if not merged_content:
        raise ValueError("No content to group")
    
    grouped_content = []
    section_text = []
    start_time, end_time = None, None

    for i, (start, end, text) in enumerate(merged_content):
        if start_time is None:
            start_time = start

        if i > 0 and (time_to_seconds(end) - time_to_seconds(start_time)) > interval:
            grouped_content.append((start_time, end_time, " ".join(section_text)))
            section_text = []
            start_time = start

        section_text.append(text)
        end_time = end

    if section_text:
        grouped_content.append((start_time, end_time, " ".join(section_text)))

    return grouped_content


def write_output_file(content: str, output_path: Path, compress: bool = False):
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Criar backup se o arquivo já existir
        if output_path.exists():
            backup_path = create_backup(output_path)
            if backup_path:
                logging.info(f"Created backup at {backup_path}")
        
        if compress:
            with tqdm(total=len(content), desc="Compressing output") as pbar:
                with gzip.open(str(output_path) + '.gz', 'wt', encoding='utf-8') as f:
                    chunk_size = 1024 * 1024
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i + chunk_size]
                        f.write(chunk)
                        pbar.update(len(chunk))
            logging.info(f"Successfully wrote compressed file to {output_path}.gz")
        else:
            with tqdm(total=len(content), desc="Writing output") as pbar:
                with output_path.open('w', encoding='utf-8') as f:
                    chunk_size = 1024 * 1024
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i + chunk_size]
                        f.write(chunk)
                        pbar.update(len(chunk))
            logging.info(f"Successfully wrote file to {output_path}")
    except Exception as e:
        logging.error(f"Error writing to {output_path}: {str(e)}")
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert VTT files to transcript",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing VTT files"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("transcript.md"),
        help="Output file path"
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=60,
        help="Time interval in seconds for grouping captions"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["markdown", "txt", "json", "html", "csv"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search for VTT files recursively in subdirectories"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log file path"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output file using gzip"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker threads for processing files"
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Custom HTML template file path"
    )
    parser.add_argument(
        "--cache-days",
        type=int,
        default=CACHE_EXPIRY_DAYS,
        help="Number of days to keep cache entries"
    )
    return parser.parse_args()


def check_memory_available():
    """Verifica se há memória suficiente disponível."""
    memory = psutil.virtual_memory()
    if memory.available < MIN_MEMORY_AVAILABLE:
        raise MemoryError(f"Insufficient memory available. Required: {MIN_MEMORY_AVAILABLE/1024/1024}MB")


def main():
    args = parse_arguments()
    setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        log_file=args.log_file
    )
    
    try:
        check_memory_available()
        
        with tqdm(total=4, desc="Overall progress") as pbar:
            vtt_files = find_vtt_files(args.input_dir, args.recursive)
            if not vtt_files:
                raise VTTProcessingError(f"No VTT files found in {args.input_dir}")
                
            merged_content = merge_vtt_files(args.input_dir, max_workers=args.workers)
            pbar.update(1)
            
            grouped_content = group_captions_by_interval(merged_content, args.interval)
            pbar.update(1)
            
            output_content = format_output(
                grouped_content, 
                args.format,
                template_path=args.template
            )
            pbar.update(1)
            
            write_output_file(output_content, args.output, compress=args.compress)
            pbar.update(1)
            
        logging.info("Processing completed successfully")
    except MemoryError as e:
        logging.error(f"Memory Error: {str(e)}")
        raise
    except VTTProcessingError as e:
        logging.error(f"VTT Processing Error: {str(e)}")
        raise
    except OutputFormatError as e:
        logging.error(f"Output Format Error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
