"""Phase 2 contract tests; all service operations and CLI actions are mocked."""
import os,sys
from pathlib import Path
os.environ.setdefault('AWS_ACCESS_KEY_ID','testing')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY','testing')
os.environ['AWS_EC2_METADATA_DISABLED']='true'
os.environ['AGENT_TRACING_ENABLED']='false'
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
from unittest.mock import Mock
import boto3
from botocore.validate import validate_parameters
import pytest
import agent_orchestrator as a
import agentcore_cli


@pytest.fixture(autouse=True)
def local_config(monkeypatch):
    monkeypatch.setattr(a.config,'_exports_cache',{})


def guardrail_client(monkeypatch, existing=False):
    client=Mock()
    client.get_paginator.return_value.paginate.return_value=[{'guardrails':
        [{'name':a.config.GUARDRAIL_NAME,'id':'abc123'}] if existing else []}]
    client.create_guardrail.return_value={'guardrailId':'abc123'}
    client.get_guardrail.return_value={'status':'READY'}
    client.create_guardrail_version.return_value={'version':'2'}
    monkeypatch.setattr(a.boto3,'client',lambda *args,**kwargs:client)
    return client


def test_guardrail_exact_policies_and_numbered_version(monkeypatch):
    shape=boto3.client('bedrock',region_name='us-east-1').meta.service_model.operation_model('CreateGuardrail').input_shape
    client=guardrail_client(monkeypatch)
    assert a.create_guardrail()==('abc123','2')
    request=client.create_guardrail.call_args.kwargs
    validate_parameters(request,shape)
    strengths={f['type']:f['inputStrength'] for f in request['contentPolicyConfig']['filtersConfig']}
    assert strengths==dict(SEXUAL='HIGH',VIOLENCE='HIGH',HATE='HIGH',INSULTS='MEDIUM',MISCONDUCT='MEDIUM')
    assert all(f['inputStrength']==f['outputStrength'] for f in request['contentPolicyConfig']['filtersConfig'])
    assert {p['type']:p['action'] for p in request['sensitiveInformationPolicyConfig']['piiEntitiesConfig']}=={
        'CREDIT_DEBIT_CARD_NUMBER':'BLOCK','US_SOCIAL_SECURITY_NUMBER':'BLOCK','EMAIL':'ANONYMIZE','PHONE':'ANONYMIZE'}
    assert request['topicPolicyConfig']['tierConfig']=={'tierName':'STANDARD'}
    assert request['crossRegionConfig']=={'guardrailProfileIdentifier':'us.guardrail.v1:0'}
    topics=request['topicPolicyConfig']['topicsConfig']
    assert len(topics)==3 and all(t['type']=='DENY' for t in topics)
    assert 'Excludes arithmetic' in topics[1]['definition']
    assert request['wordPolicyConfig']['managedWordListsConfig']==[{'type':'PROFANITY'}]
    assert client.get_guardrail.call_args.kwargs['guardrailVersion']=='2'


def test_existing_guardrail_updated_then_published(monkeypatch):
    client=guardrail_client(monkeypatch,True)
    assert a.create_guardrail()==('abc123','2')
    client.create_guardrail.assert_not_called()
    client.update_guardrail.assert_called_once()
    client.create_guardrail_version.assert_called_once()


def test_guardrail_rejects_draft_version_and_failure(monkeypatch):
    client=guardrail_client(monkeypatch)
    client.create_guardrail_version.return_value={'version':'DRAFT'}
    with pytest.raises(RuntimeError,match='numbered'):
        a.create_guardrail()
    client.get_guardrail.return_value={'status':'FAILED'}
    with pytest.raises(RuntimeError,match='failed'):
        a.create_guardrail()


def memory_client(monkeypatch):
    client=Mock()
    client.get_paginator.return_value.paginate.return_value=[{'memories':[]}]
    client.create_memory.return_value={'memory':{'id':a.config.MEMORY_NAME+'-123','arn':'arn:memory','status':'CREATING'}}
    client.get_memory.return_value={'memory':{'id':a.config.MEMORY_NAME+'-123','arn':'arn:memory','status':'ACTIVE'}}
    monkeypatch.setattr(a,'agentcore_control',client)
    monkeypatch.setattr(a.time,'sleep',lambda seconds:None)
    return client


def test_memory_request_and_wait_until_active(monkeypatch):
    shape=boto3.client('bedrock-agentcore-control',region_name='us-east-1').meta.service_model.operation_model('CreateMemory').input_shape
    client=memory_client(monkeypatch)
    assert a.configure_memory('arn:runtime')=='arn:memory'
    request=client.create_memory.call_args.kwargs
    validate_parameters(request,shape)
    assert request['eventExpiryDuration']==7
    assert request['memoryStrategies']==[{'summaryMemoryStrategy':{'name':'SessionSummary','namespaces':['/summaries/{actorId}/{sessionId}']}}]
    client.get_memory.assert_called_once()


def test_memory_failure_and_timeout_do_not_report_success(monkeypatch):
    client=memory_client(monkeypatch)
    client.get_memory.return_value={'memory':{'id':'m','arn':'arn:memory','status':'FAILED','failureReason':'failed strategy'}}
    with pytest.raises(RuntimeError,match='failed strategy'):
        a.configure_memory('arn:runtime')
    ticks=iter([0,301])
    monkeypatch.setattr(a.time,'monotonic',lambda:next(ticks))
    with pytest.raises(TimeoutError):
        a.configure_memory('arn:runtime')


def test_existing_memory_verified_and_reused(monkeypatch):
    client=memory_client(monkeypatch)
    client.get_paginator.return_value.paginate.return_value=[{'memories':[{'id':a.config.MEMORY_NAME+'-123'}]}]
    client.get_memory.return_value={'memory':{'id':'m','arn':'arn:memory','status':'ACTIVE','eventExpiryDuration':7,
        'strategies':[{'type':'SUMMARIZATION','namespaces':['/summaries/{actorId}/{sessionId}']}]}}
    assert a.configure_memory('arn:runtime')=='arn:memory'
    client.create_memory.assert_not_called()


def test_runtime_uses_cli_with_all_required_configuration(monkeypatch):
    for key,value in {'ACCOUNT_ID':'123456789012','AGENTCORE_ROLE_ARN':'arn:role',
        'RETURNS_KB_ID':'RETURNS','SHIPPING_KB_ID':'SHIPPING','WARRANTY_KB_ID':'WARRANTY',
        'AGENT_LOG_GROUP':'/aws/bedrock/agentcore/test'}.items():
        monkeypatch.setitem(a.config.__dict__,key,value)
    for name in ['stage_runtime_code','configure_runtime','deploy']:
        monkeypatch.setattr(agentcore_cli,name,Mock())
    monkeypatch.setattr(agentcore_cli,'deployed_runtime_arn',Mock(side_effect=['','arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test']))
    ready=Mock();monkeypatch.setattr(a,'wait_for_runtime_ready',ready)
    arn=a.deploy_to_agentcore_runtime(None,'abc123','2')
    cfg=agentcore_cli.configure_runtime.call_args.kwargs
    assert cfg['network_mode']=='PUBLIC' and cfg['protocol']=='HTTP' and cfg['execution_role_arn']=='arn:role'
    assert set(cfg['env_vars'])=={'AWS_REGION','PROJECT_NAME','RETURNS_KB_ID','SHIPPING_KB_ID',
        'WARRANTY_KB_ID','AGENT_LOG_GROUP','GUARDRAIL_ID','GUARDRAIL_VERSION'}
    assert cfg['env_vars']['GUARDRAIL_VERSION']=='2'
    agentcore_cli.stage_runtime_code.assert_called_once()
    agentcore_cli.deploy.assert_called_once()
    ready.assert_called_once_with(a.agentcore_control,'test')
    assert arn.endswith('/test')


@pytest.mark.parametrize('missing', ['guardrail','kb'])
def test_runtime_preconditions_prevent_cli_deployment(monkeypatch,missing):
    for key,value in {'ACCOUNT_ID':'123456789012','RETURNS_KB_ID':'R','SHIPPING_KB_ID':'S',
        'WARRANTY_KB_ID':'' if missing=='kb' else 'W','AGENT_LOG_GROUP':'logs'}.items():
        monkeypatch.setitem(a.config.__dict__,key,value)
    monkeypatch.setattr(agentcore_cli,'stage_runtime_code',Mock())
    monkeypatch.setattr(agentcore_cli,'configure_runtime',Mock())
    monkeypatch.setattr(agentcore_cli,'deploy',Mock())
    monkeypatch.setattr(agentcore_cli,'deployed_runtime_arn',Mock(return_value=''))
    with pytest.raises(ValueError):
        a.deploy_to_agentcore_runtime(None,'g','DRAFT' if missing=='guardrail' else '1')
    agentcore_cli.stage_runtime_code.assert_not_called()
    agentcore_cli.configure_runtime.assert_not_called()
    agentcore_cli.deploy.assert_not_called()
