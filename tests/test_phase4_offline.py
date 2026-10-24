"""Observability configuration contract, without external mutations."""
from unittest.mock import Mock
import pytest
from test_phase2_offline import a, local_config


def test_observability_enables_info_and_full_sampling(monkeypatch):
    monkeypatch.setitem(a.config.__dict__,'AGENT_LOG_GROUP','/aws/bedrock/agentcore/test')
    apply=Mock(return_value={'log_group':'/aws/bedrock/agentcore/test'})
    monkeypatch.setattr(a,'apply_observability_config',apply)
    a.configure_observability('arn:runtime')
    apply.assert_called_once_with('arn:runtime',{
        'cloudWatchConfig':{'logGroupName':'/aws/bedrock/agentcore/test','logLevel':'INFO','enabled':True},
        'xRayConfig':{'enabled':True,'samplingRate':1.0}})


def test_observability_failure_is_not_reported_as_deployment_success(monkeypatch,capsys):
    monkeypatch.setitem(a.config.__dict__,'AGENT_LOG_GROUP','logs')
    monkeypatch.setattr(a,'apply_observability_config',Mock(side_effect=RuntimeError('denied')))
    with pytest.raises(RuntimeError,match='denied'):
        a.configure_observability('arn:runtime')
    assert '[Note] Observability configuration failed: denied' in capsys.readouterr().out
