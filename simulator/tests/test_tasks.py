"""Task-loader tests: Online-Mind2Web loading + load_tasks source routing (no network)."""

import json

import pytest

from simulator.tasks import load_online_mind2web_tasks

OM2W_ROWS = [
	{
		'task_id': 'b7258ee05d75e6c50673a59914db412e_110325',
		'confirmed_task': "Find the store location and hours of the closest Trader Joe's to zip code 90028.",
		'website': 'https://www.traderjoes.com/',
		'reference_length': 6,
		'level': 'medium',
	},
	{
		'task_id': '92a3d4236f167af4afdc08876a902ba6',
		'confirmed_task': 'Find a 2022 Tesla Model 3 on CarMax.',
		'website': 'https://www.carmax.com/',
		'reference_length': 10,
		'level': 'easy',
	},
]


@pytest.fixture
def om2w_file(tmp_path):
	p = tmp_path / 'online_mind2web.json'
	p.write_text(json.dumps(OM2W_ROWS))
	return p


def test_load_online_mind2web_tasks(om2w_file):
	tasks = load_online_mind2web_tasks(om2w_file)
	assert len(tasks) == 2
	t = tasks[0]
	assert t.source == 'online_mind2web'
	assert t.id == 'b7258ee05d75e6c50673a59914db412e_110325'
	assert t.site == 'traderjoes.com'  # hostname, www stripped
	assert t.start_url == 'https://www.traderjoes.com/'
	assert t.question.startswith('Find the store location')
	assert t.level == 'medium'
	assert t.reference_length == 6
	assert t.reference_answer is None  # no reference answers — WebJudge protocol
	assert t.folder_name.startswith('online_mind2web__')
	assert '/' not in t.folder_name


def test_load_online_mind2web_missing_file(tmp_path):
	with pytest.raises(SystemExit, match='download_data'):
		load_online_mind2web_tasks(tmp_path / 'nope.json')
