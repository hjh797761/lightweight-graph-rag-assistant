"""Structure evidence must be recoverable from source positions, never guessed."""

import importlib
from zipfile import ZipFile

import pytest


def write_pdf_fixture(path, pages):
    """A real PDF with text streams and xref; no native library needed to create it."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
        ("<< /Type /Pages /Count %d /Kids [%s] >>" % (len(pages), " ".join(f"{4 + 2*i} 0 R" for i in range(len(pages))))).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for index, text in enumerate(pages):
        escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in text.splitlines()]
        stream = ("BT /F1 12 Tf 72 720 Td 14 TL " + " T* ".join(f"({line}) Tj" for line in escaped) + " ET").encode("ascii")
        objects.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (5 + 2*index)).encode())
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(data)


def structure():
    assert importlib.util.find_spec("graphrag.structure") is not None, "structure parser is missing"
    return importlib.import_module("graphrag.structure")


def test_repeated_titles_have_position_based_ids_and_true_hierarchy():
    parser = structure()
    text = "# 范围\n甲。\n## 子节\n乙。\n# 范围\n丙。"
    chunks = parser.parse_text(text, "doc")
    assert len({c.section_id for c in chunks}) == 3
    assert [c.title_path for c in chunks] == [("范围",), ("范围", "子节"), ("范围",)]
    assert [c.id for c in chunks] == [c.id for c in parser.parse_text(text, "doc")]
    assert all(c.source_text == text[c.source_start:c.source_end] for c in chunks)
    assert chunks[1].locator == "lines 3-4"


def test_sentence_splitting_preserves_original_offsets_and_only_cuts_long_sentences():
    parser = structure()
    text = "# H\r\n甲甲甲。乙乙乙。\r\n\r\n" + "长" * 13 + "。"
    chunks = parser.parse_text(text, "doc", chunk_size=8)
    assert all(c.source_text == text[c.source_start:c.source_end] for c in chunks)
    assert any("甲甲甲。" in c.source_text for c in chunks)
    assert any("乙乙乙。" in c.source_text for c in chunks)
    assert all(len(c.source_text) <= 8 for c in chunks)
    assert all(c.partial == ("长" in c.source_text) for c in chunks)
    assert "".join(c.source_text for c in chunks).replace("\r", "").replace("\n", "") == text.replace("\r", "").replace("\n", "")


def test_sectionless_text_has_no_invented_membership_or_adjacency():
    parser = structure()
    chunks = parser.parse_text("Plain heading\n\nFirst paragraph.\n\nSecond paragraph.", "doc", format="txt", chunk_size=20)
    assert all(c.section_id is None and c.title_path == () for c in chunks)
    assert parser.build_links(chunks) == []


def test_markdown_code_fences_are_not_headings():
    chunks = structure().parse_text("```md\n# Example\n```\n\nbody", "doc")
    assert all(c.section_id is None for c in chunks)


def test_markdown_setext_headings_follow_declared_hierarchy():
    chunks = structure().parse_text("Main\n====\nbody\n\nChild\n-----\nchild body\n\n# Next\nend", "doc")
    assert [c.title_path for c in chunks] == [("Main",), ("Main", "Child"), ("Next",)]


def test_long_sentence_reference_can_cross_a_forced_chunk_boundary():
    parser = structure()
    text = "# 第一条\n" + "甲" * 7 + "第三条" + "乙" * 10 + "。\n# 第三条\n正文。"
    chunks = parser.parse_text(text, "doc", chunk_size=8)
    refs = [link for link in parser.build_links(chunks) if link.kind == "explicit_reference"]
    assert len(refs) == 1
    assert refs[0].status == "resolved" and refs[0].basis == "第三条"
    assert text[refs[0].source_start:refs[0].source_end] == refs[0].basis


@pytest.mark.parametrize("heading", ["###### 第1条", "   ###### 第1条 ###", "###### Clause 1. See section 2.1 ###"])
def test_forced_heading_cuts_never_turn_declarations_into_references(heading):
    parser = structure()
    text = heading + "\nbody."
    for size in (4, 800):
        chunks = parser.parse_text(text, "doc", chunk_size=size)
        refs = [link for link in parser.build_links(chunks) if link.kind == "explicit_reference"]
        assert refs == []


def test_unsupported_english_identifier_suffix_never_resolves_a_prefix():
    parser = structure()
    text = "# Section 2\nRule.\n# Section 2.1a\nOther rule.\n# Notes\nSee section2.1a."
    chunks = parser.parse_text(text, "doc")
    refs = [link for link in parser.build_links(chunks) if link.kind == "explicit_reference"]
    assert refs == []
    text = "# Section 2.1a\nOther rule.\n# Notes\nSee section2."
    chunks = parser.parse_text(text, "doc")
    refs = [link for link in parser.build_links(chunks) if link.kind == "explicit_reference"]
    assert len(refs) == 1 and refs[0].status == "missing"


@pytest.mark.parametrize("gap", ["\n", "\r\n", " \n\t "])
def test_reference_retains_exact_whitespace_across_chunk_boundaries(gap):
    parser = structure()
    text = "# Section 2 Rules\nActual section.\n# Notes\nSee section" + gap + "2 for the exception."
    for size in (20, 800):
        chunks = parser.parse_text(text, "doc", chunk_size=size)
        for previous, chunk in zip(chunks, chunks[1:]):
            if previous.section_id == chunk.section_id:
                assert chunk.source_gap_before == text[previous.source_end:chunk.source_start]
        refs = [link for link in parser.build_links(chunks) if link.kind == "explicit_reference"]
        assert len(refs) == 1 and refs[0].status == "resolved"
        assert refs[0].basis == "section" + gap + "2"
        assert text[refs[0].source_start:refs[0].source_end] == refs[0].basis


def test_numbered_txt_sections_resolve_forward_and_missing_references():
    parser = structure()
    text = "第一条 范围\n参见第三条及第九条。\n第三条 实施\n具体办法。"
    chunks = parser.parse_text(text, "doc", format="txt")
    links = [link for link in parser.build_links(chunks) if link.kind == "explicit_reference"]
    assert len(links) == 2
    resolved = next(link for link in links if link.basis == "第三条")
    missing = next(link for link in links if link.basis == "第九条")
    assert resolved.status == "resolved"
    assert resolved.target_id == chunks[-1].id and resolved.target_ids == (chunks[-1].id,)
    assert missing.status == "missing" and missing.target_id is None and missing.target_ids == ()
    assert all(text[link.source_start:link.source_end] == link.basis for link in links)


def test_duplicate_numbered_labels_are_ambiguous_and_do_not_cross_documents():
    parser = structure()
    a = parser.parse_text("# 第1节 概述\n参见第2节。\n# 第2节 一\n甲。\n# 第2节 二\n乙。", "a")
    b = parser.parse_text("# 第2节 独立\n丙。", "b")
    refs = [link for link in parser.build_links(a + b) if link.kind == "explicit_reference"]
    assert len(refs) == 1
    assert refs[0].status == "ambiguous" and refs[0].target_id is None
    c = parser.parse_text("# Start\nSee section 2.1.", "c")
    d = parser.parse_text("# Section 2.1 Rules\nOther document.", "d")
    ref = next(link for link in parser.build_links(c + d) if link.kind == "explicit_reference")
    assert ref.status == "missing"


def test_multichunk_target_and_same_section_adjacency():
    parser = structure()
    text = "# Section 1 Intro\nSee section2.1.\n# Section 2.1 Rules\nAlpha sentence.\n\nBeta sentence.\n# Other\nEnd."
    chunks = parser.parse_text(text, "doc", chunk_size=24)
    links = parser.build_links(chunks)
    ref = next(link for link in links if link.kind == "explicit_reference")
    targets = [c.id for c in chunks if c.title_path == ("Section 2.1 Rules",)]
    assert len(targets) >= 2 and ref.target_ids == tuple(targets)
    by_id = {c.id: c for c in chunks}
    adjacent = [link for link in links if link.kind == "adjacent"]
    assert adjacent
    assert all(by_id[x.source_id].section_id == by_id[x.target_id].section_id for x in adjacent)
    assert all(abs(by_id[x.target_id].sequence - by_id[x.source_id].sequence) == 1 for x in adjacent)
    assert {(x.source_id, x.target_id) for x in adjacent} == {(x.target_id, x.source_id) for x in adjacent}
    assert len([link for link in links if link.kind == "section_member"]) == len(chunks)


def test_txt_does_not_promote_plain_list_items_to_sections():
    chunks = structure().parse_text("Shopping\n1. apples\n2. pears", "doc", format="txt")
    assert all(c.section_id is None for c in chunks)


def test_real_docx_uses_heading_styles_and_paragraph_locations(tmp_path):
    parser = structure()
    from docx import Document

    path = tmp_path / "source.docx"
    # A real minimal package avoids dependence on the optional default.docx template.
    with ZipFile(path, "w") as package:
        package.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
        package.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        package.writestr("word/_rels/document.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        package.writestr("word/styles.xml", '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/></w:style></w:styles>')
        package.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>范围</w:t></w:r></w:p><w:p><w:r><w:t># This is ordinary text</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>细则</w:t></w:r></w:p><w:p><w:r><w:t>Content.</w:t></w:r></w:p></w:body></w:document>')
    chunks = parser.parse_file(path, "doc")
    assert [c.title_path for c in chunks] == [("范围",), ("范围", "细则")]
    assert [c.locator for c in chunks] == ["paragraphs 1-2", "paragraphs 3-4"]
    source = "\n".join(p.text for p in Document(path).paragraphs)
    assert all(c.source_text == source[c.source_start:c.source_end] for c in chunks)


def test_real_pdf_uses_pages_without_inventing_sections(tmp_path):
    parser = structure()
    path = tmp_path / "source.pdf"
    write_pdf_fixture(path, ["# Looks like a heading\nFirst page.\nSee section", "2. Second page."])
    chunks = parser.parse_file(path, "doc")
    assert [c.locator for c in chunks] == ["page 1", "page 2"]
    assert all(c.section_id is None for c in chunks)
    assert parser.build_links(chunks) == []
    from PyPDF2 import PdfReader
    with path.open("rb") as stream:
        source = "\n".join(page.extract_text() or "" for page in PdfReader(stream).pages)
    assert all(c.source_text == source[c.source_start:c.source_end] for c in chunks)


def test_text_file_retains_crlf_and_reports_lines(tmp_path):
    path = tmp_path / "source.md"
    source = "# Heading\r\nContent.\r\n"
    path.write_bytes(source.encode("utf-8"))
    chunks = structure().parse_file(path, "doc")
    assert chunks[0].source_text == source.rstrip()
    assert chunks[0].locator == "lines 1-2"


def test_parser_rejects_invalid_limits_and_formats():
    parser = structure()
    with pytest.raises(ValueError, match="chunk_size"):
        parser.parse_text("body", "doc", chunk_size=0)
    with pytest.raises(ValueError, match="format"):
        parser.parse_text("body", "doc", format="html")
