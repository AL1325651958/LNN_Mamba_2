"""
Extract the longest continuous non-zero Power segments from wind/solar multimodal data.
Outputs each segment as a separate CSV file to the root directory.

Strategy:
- Power > 0 defines "valid" data (station is generating)
- Merge nearby non-zero segments across small zero gaps to minimize file count
- Adaptive gap tolerance: larger for stations with many scattered zeros
- Each merged segment -> one CSV file
"""

import pandas as pd
import numpy as np
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_DIR = r'f:\Code\SCI\LNN_Mamba_2\17304221\data_processed'
OUT_DIR = r'f:\Code\SCI\LNN_Mamba_2'

# Minimum segment lengths (in 15-min intervals) - filter out tiny fragments
MIN_WIND_SEGMENT = 96   # 24 hours
MIN_SOLAR_SEGMENT = 64  # 16 hours


def read_file(path):
    """Read xlsx with auto-detecting sheet name."""
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)


def sanitize_filename(name):
    """Remove problematic characters for filenames."""
    return name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_')


def fix_time_column(df, time_col):
    """Fix time column, handling 24:00:00 format."""
    # Replace "24:00:00" with "00:00:00" of next day
    time_str = df[time_col].astype(str)
    mask_24 = time_str.str.contains('24:00:00', na=False)
    if mask_24.any():
        print(f'  Fixed {mask_24.sum()} rows with 24:00:00 time format')
        # Replace 24:00:00 with 00:00:00 and add 1 day
        fixed = time_str.str.replace('24:00:00', '00:00:00')
        df[time_col] = pd.to_datetime(fixed) + pd.to_timedelta(mask_24.astype(int), unit='D')
    else:
        df[time_col] = pd.to_datetime(df[time_col])
    return df


def find_nonzero_segments(is_nonzero):
    """
    Find all contiguous segments where condition is True.
    Returns list of (start_idx, end_idx, length) tuples.
    """
    segments = []
    i = 0
    n = len(is_nonzero)
    while i < n:
        if is_nonzero[i]:
            j = i
            while j < n and is_nonzero[j]:
                j += 1
            segments.append((i, j - 1, j - i))
            i = j
        else:
            i += 1
    return segments


def merge_nearby_segments(segments, gap_tolerance):
    """
    Merge segments separated by short zero-gaps.
    gap_tolerance: max gap in 15-min intervals allowed between merged segments.

    The merged segment includes the zero-gap data (for continuity).
    Returns merged segments sorted by length descending.
    """
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg[0] - prev[1] - 1
        if gap <= gap_tolerance:
            # Merge: include the zero gap in the output
            merged[-1] = (prev[0], seg[1], seg[1] - prev[0] + 1)
        else:
            merged.append(seg)

    # Sort by length descending
    merged.sort(key=lambda x: -x[2])
    return merged


def process_file(filepath, category):
    """Process one xlsx file and return its non-zero segments."""
    try:
        df = read_file(filepath)
    except Exception as e:
        print(f'  ERROR reading {filepath}: {e}')
        return None, None, [], None

    time_col = df.columns[0]

    # Find the Power column
    power_cols = [c for c in df.columns if 'power' in c.lower()]
    if not power_cols:
        power_col = df.columns[-1]
        print(f'  WARNING: No Power column found, using "{power_col}"')
    else:
        power_col = power_cols[0]

    # Fix and parse time
    df = fix_time_column(df, time_col)

    # Find raw non-zero segments (Power > 0)
    is_nonzero = (df[power_col] > 0).values
    raw_segments = find_nonzero_segments(is_nonzero)

    if not raw_segments:
        print(f'  No non-zero Power segments found!')
        return df, time_col, [], power_col

    # Calculate zero density to determine merge strategy
    zero_ratio = 1 - sum(s[2] for s in raw_segments) / len(df)
    print(f'  Raw segments: {len(raw_segments)}, zero_ratio: {zero_ratio:.1%}')

    # Adaptive gap tolerance based on data characteristics
    if category == 'solar':
        # Solar: nighttime zeros, merge across nights (24h window)
        # This combines consecutive days into multi-day blocks
        gap_tolerance = 96  # 24 hours
    else:
        # Wind: adapt based on zero density
        if zero_ratio < 0.02:
            gap_tolerance = 48   # 12h - nearly continuous data
        elif zero_ratio < 0.10:
            gap_tolerance = 96   # 24h - moderate zeros
        elif zero_ratio < 0.25:
            gap_tolerance = 192  # 48h - many small zero gaps
        else:
            gap_tolerance = 384  # 96h - very fragmented

    print(f'  Gap tolerance: {gap_tolerance} intervals ({gap_tolerance*15/60:.0f}h)')

    merged = merge_nearby_segments(raw_segments, gap_tolerance)
    print(f'  After merge: {len(merged)} segments')

    return df, time_col, merged, power_col


def main():
    total_segments_exported = 0
    total_pts_exported = 0
    summary_lines = []

    for root, dirs, files in os.walk(DATA_DIR):
        category = 'wind' if 'wind' in root.lower() else 'solar'
        for f in sorted(files):
            if f.endswith('.xlsx') and not f.startswith('~$'):
                filepath = os.path.join(root, f)
                basename = os.path.splitext(f)[0]
                safe_name = sanitize_filename(basename)
                print(f'\n{"="*60}')
                print(f'File: {f} [{category}]')

                df, time_col, segments, power_col = process_file(filepath, category)
                if df is None or not segments:
                    continue

                # Calculate statistics
                total_nonzero = sum(s[2] for s in segments)
                total_rows = len(df)

                # Filter by minimum length
                min_len = MIN_WIND_SEGMENT if category == 'wind' else MIN_SOLAR_SEGMENT
                qualified = [s for s in segments if s[2] >= min_len]

                if not qualified:
                    # At least export the longest segment
                    qualified = segments[:1]
                    print(f'  No segments meet minimum length, taking longest')

                # Export all qualified segments (already sorted by length)
                accumulated = 0
                accumulated_clean = 0
                prev_exported = total_segments_exported
                for rank, (start, end, length) in enumerate(qualified):
                    segment_df = df.iloc[start:end+1].copy()

                    # CRITICAL: Filter out rows where Power == 0
                    # Keep only truly non-zero data within the merged time window
                    segment_clean = segment_df[segment_df[power_col] > 0].copy()
                    clean_len = len(segment_clean)

                    if clean_len == 0:
                        print(f'  [{rank+1}] SKIPPED (all zero after filtering)')
                        continue

                    # Skip segments that are too short after zero filtering
                    if clean_len < min_len:
                        print(f'  [{rank+1}] SKIPPED (only {clean_len} clean pts, min {min_len})')
                        continue

                    merged_hours = length * 15 / 60
                    clean_hours = clean_len * 15 / 60
                    accumulated += length
                    accumulated_clean += clean_len

                    t_start = segment_clean[time_col].iloc[0].strftime('%Y%m%d')
                    t_end = segment_clean[time_col].iloc[-1].strftime('%Y%m%d')
                    zero_pct = (1 - clean_len / length) * 100

                    # Shorten name for readability
                    short_name = safe_name.replace(' (Nominal capacity-', '_').replace('MW)', 'MW').replace(' ', '_')
                    out_name = (f'{short_name}_seg{rank+1:02d}_{t_start}_{t_end}_'
                                f'{clean_len}pts_{clean_hours:.0f}h_clean.csv')
                    out_path = os.path.join(OUT_DIR, out_name)

                    segment_clean.to_csv(out_path, index=False, encoding='utf-8-sig')
                    print(f'  [{rank+1}] {out_name}  (merged {merged_hours:.0f}h, {zero_pct:.0f}% zeros removed)')
                    total_segments_exported += 1

                # Summary
                nonzero_in_original = (df[power_col] > 0).sum()
                total_h = nonzero_in_original * 15 / 60
                cov_h = accumulated_clean * 15 / 60
                pct = accumulated_clean / nonzero_in_original * 100 if nonzero_in_original > 0 else 0
                zeros_in_output = accumulated - accumulated_clean
                n_exported = total_segments_exported - prev_exported
                line = (f'{f}: {nonzero_in_original} original nonzero pts ({total_h:.0f}h), '
                        f'{len(segments)} merged segs -> {n_exported} exported, '
                        f'{accumulated_clean} clean pts ({cov_h:.0f}h, {pct:.0f}%), '
                        f'{zeros_in_output} zeros filtered')
                print(f'  SUMMARY: {line}')
                summary_lines.append(line)
                total_pts_exported += accumulated_clean

    # Write summary
    summary_path = os.path.join(OUT_DIR, 'extraction_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as fh:
        fh.write('Non-Zero Continuous Segment Extraction Summary\n')
        fh.write('=' * 60 + '\n\n')
        fh.write(f'Total CSV files exported: {total_segments_exported}\n')
        fh.write(f'Total data points exported: {total_pts_exported}\n\n')
        for line in summary_lines:
            fh.write(line + '\n')

    print(f'\n{"="*60}')
    print(f'DONE: {total_segments_exported} CSV files in {OUT_DIR}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
