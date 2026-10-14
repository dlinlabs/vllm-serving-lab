import xml.etree.ElementTree as ET

import pytest

from scripts.run_linux_cpu_tests import TRACE_CASES, validate_report


@pytest.fixture
def report(tmp_path):
    root = ET.Element('testsuites')
    suite = ET.SubElement(root, 'testsuite')
    for name in sorted(TRACE_CASES):
        ET.SubElement(suite, 'testcase', classname='tests.test_trace_replay', name=name)
    ET.SubElement(suite, 'testcase', classname='tests.test_budget_pilot',
                  name='test_full_runner_with_mock_vllm')
    path = tmp_path / 'junit.xml'

    def write():
        ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
        return path
    return suite, write


def test_acceptance_requires_complete_successful_matrix(report):
    suite, write = report
    result = validate_report(write())
    assert result['test_cases'] == len(suite) == 6
    assert result['trace_configurations'] == sorted(TRACE_CASES)


@pytest.mark.parametrize('outcome', ['failure', 'error', 'skipped'])
@pytest.mark.parametrize('index', [0, -1], ids=['trace', 'other-test'])
def test_acceptance_rejects_unsuccessful_cases(report, outcome, index):
    suite, write = report
    ET.SubElement(suite[index], outcome)
    with pytest.raises(RuntimeError, match='failed, errored or skipped'):
        validate_report(write())


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong-module', 'wrong-configuration', 'empty'])
def test_acceptance_rejects_incomplete_or_ambiguous_matrix(report, damage):
    suite, write = report
    if damage == 'missing':
        suite.remove(suite[0])
    elif damage == 'duplicate':
        ET.SubElement(suite, 'testcase', **suite[0].attrib)
    elif damage == 'wrong-module':
        suite[0].set('classname', 'tests.not_trace_replay')
    elif damage == 'wrong-configuration':
        suite[0].set('name', 'test_full_trace_runner_and_adversarial_analysis[unexpected]')
    else:
        suite.clear()
    with pytest.raises(RuntimeError, match='five trace configurations exactly once'):
        validate_report(write())


def test_acceptance_requires_a_parseable_report(tmp_path):
    path = tmp_path / 'junit.xml'
    with pytest.raises(FileNotFoundError):
        validate_report(path)
    path.write_text('<testsuites>', encoding='utf-8')
    with pytest.raises(ET.ParseError):
        validate_report(path)
