"""Offline behavior checks. Moto emulates DynamoDB; no AWS calls or model inference."""
import os
os.environ.setdefault('AWS_ACCESS_KEY_ID', 'testing')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'testing')
os.environ.setdefault('AWS_EC2_METADATA_DISABLED', 'true')
os.environ['AGENT_TRACING_ENABLED'] = 'false'
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'src')]
from datetime import datetime, timedelta, timezone
from threading import Barrier
from unittest.mock import Mock
import boto3
import pytest
from moto import mock_aws
import agent_orchestrator as a


@pytest.fixture
def db(monkeypatch):
    with mock_aws():
        resource = boto3.resource('dynamodb', region_name='us-east-1')
        for name, keys in [('orders', ['customer_id', 'order_id']), ('customers', ['customer_id']), ('workflow', ['session_id'])]:
            resource.create_table(TableName=name, BillingMode='PAY_PER_REQUEST',
                AttributeDefinitions=[{'AttributeName':k,'AttributeType':'S'} for k in keys],
                KeySchema=[{'AttributeName':k,'KeyType':'HASH' if i==0 else 'RANGE'} for i,k in enumerate(keys)])
        monkeypatch.setattr(a, 'dynamodb', resource)
        monkeypatch.setattr(a.config, '_exports_cache', {})
        for key,val in [('ORDERS_TABLE','orders'),('CUSTOMERS_TABLE','customers'),('WORKFLOW_STATE_TABLE','workflow')]:
            monkeypatch.setitem(a.config.__dict__,key,val)
        monkeypatch.setattr(a, 'trace', Mock())
        yield resource


def function(agent, name):
    return agent.tool_registry.registry[name]._tool_func


def setup_order(db, tier='Standard', days=10, delivered=True):
    db.Table('customers').put_item(Item={'customer_id':'C1','tier':tier})
    item={'customer_id':'C1','order_id':'O1','status':'delivered','product_category':'Electronics',
          'order_date':'2026-01-01'}
    if delivered:
        item['delivery_date']=(datetime.now(timezone.utc)-timedelta(days=days)).date().isoformat()
    db.Table('orders').put_item(Item=item)
    a._create_workflow_state('S1','C1')
    a._update_workflow_state('S1',{'inventory_agent':'verified inventory'},0)


def test_model_assignments_and_tool_counts(db):
    agents=[a.build_inventory_agent(),a.build_refund_agent(),a.build_policy_agent(),a.build_communication_agent()]
    root=a.build_orchestrator_agent(*agents)
    assert [len(x.tool_registry.registry) for x in agents+[root]] == [3,2,1,1,5]
    assert root.model.config['model_id']==a.config.ORCHESTRATOR_MODEL_ID
    assert all(x.model.config['model_id']==a.config.WORKER_MODEL_ID for x in agents)


def test_version_and_disjoint_concurrent_updates(db):
    a._create_workflow_state('S1','C1')
    first=a._update_workflow_state('S1',{'inventory_agent':'facts'},0)
    second=a._update_workflow_state('S1',{'policy_agent':'policy'},0)
    assert first['version']==1 and second['version']==2
    assert second['inventory_agent']=='facts' and second['policy_agent']=='policy'


def test_stale_same_field_cannot_overwrite(db):
    a._create_workflow_state('S1','C1')
    a._update_workflow_state('S1',{'inventory_agent':'new facts'},0)
    with pytest.raises(RuntimeError,match='same agent'):
        a._update_workflow_state('S1',{'inventory_agent':'stale facts'},0)
    assert a._read_workflow_state('S1')['inventory_agent']=='new facts'


def test_missing_session_not_recreated_and_metadata_protected(db):
    with pytest.raises(RuntimeError,match='deleted or expired'):
        a._update_workflow_state('missing',{'inventory_agent':'facts'},0)
    with pytest.raises(ValueError):
        a._update_workflow_state('S1',{'customer_id':'C2'},0)


@pytest.mark.parametrize('tier,days,expected', [('Standard',30,'return_initiated'),('Standard',31,'ineligible'),
    ('Premium',60,'return_initiated'),('Premium',61,'ineligible')])
def test_return_window_boundaries(db,tier,days,expected):
    setup_order(db,tier,days)
    token=a._worker_scope.set({'session_id':'S1','customer_id':'C1'})
    try:
        tool=function(a.build_refund_agent(),'initiate_refund')
        result=tool('C1','O1','No longer needed')
        assert result['status']==expected
        if expected=='return_initiated':
            assert tool('C1','O1','retry')['return_reference']==result['return_reference']
    finally:
        a._worker_scope.reset(token)


def test_missing_delivery_date_and_customer_scope(db):
    setup_order(db,delivered=False)
    token=a._worker_scope.set({'session_id':'S1','customer_id':'C1'})
    try:
        tool=function(a.build_refund_agent(),'initiate_refund')
        assert tool('C1','O1','reason')['status']=='review_required'
        with pytest.raises(PermissionError):
            tool('C2','O1','reason')
        assert db.Table('orders').get_item(Key={'customer_id':'C1','order_id':'O1'})['Item']['status']=='delivered'
    finally:
        a._worker_scope.reset(token)


def test_initialization_resume_and_ownership(db):
    root=a.build_orchestrator_agent(None,None,None,None)
    initialize=function(root,'initialize_session')
    initialize('S1','C1')
    a._update_workflow_state('S1',{'communication_agent':'old reply'},0)
    initialize('S1','C1')
    state=a._read_workflow_state('S1')
    assert state['communication_agent']==''
    assert state['previous_turn']['communication_agent']=='old reply'
    with pytest.raises(PermissionError):
        initialize('S1','C2')


def test_route_enforces_inventory_then_persists_results(db,monkeypatch):
    root=a.build_orchestrator_agent(lambda prompt:'order facts', lambda prompt:'review required',
        None, lambda prompt:'Please confirm delivery date.')
    function(root,'initialize_session')('S1','C1')
    with pytest.raises(ValueError,match='Inventory'):
        function(root,'route_to_refund_agent')('S1','C1','return O1')
    monkeypatch.setattr(a,'build_inventory_agent',lambda: lambda prompt:'order facts')
    monkeypatch.setattr(a,'build_refund_agent',lambda: lambda prompt:'review required')
    monkeypatch.setattr(a,'build_communication_agent',lambda: lambda prompt:'Please confirm delivery date.')
    function(root,'route_to_inventory_agent')('S1','C1','return O1')
    function(root,'route_to_refund_agent')('S1','C1','return O1')
    result=function(root,'route_to_communication_agent')('S1','C1','return O1')
    state=a._read_workflow_state('S1')
    assert state['version']==3
    assert state['communication_agent']==result


def test_three_retrievers_are_concurrent_and_isolated_failures(db,monkeypatch):
    monkeypatch.setattr(a.config,'RETURNS_KB_ID','R',raising=False)
    monkeypatch.setattr(a.config,'SHIPPING_KB_ID','S',raising=False)
    monkeypatch.setattr(a.config,'WARRANTY_KB_ID','W',raising=False)
    original_new=a._new_agent
    barrier=Barrier(3,timeout=5)
    def new_agent(name,*args,**kwargs):
        if name == 'PolicyAgent':
            return original_new(name,*args,**kwargs)
        def run(query):
            barrier.wait()
            if name.startswith('Shipping'):
                raise RuntimeError('simulated retrieval outage')
            return name+' cited evidence'
        return run
    monkeypatch.setattr(a,'_new_agent',new_agent)
    policy=a.build_policy_agent()
    result=function(policy,'search_all_policies')('return and warranty')
    assert 'ReturnsPolicyRetrieverAgent cited evidence' in result
    assert 'WarrantyPolicyRetrieverAgent cited evidence' in result
    assert '[Shipping]\nUNAVAILABLE' in result


def test_inventory_lookup_and_query_stay_with_scoped_customer(db):
    setup_order(db)
    db.Table('orders').put_item(Item={'customer_id':'C2','order_id':'OTHER','status':'shipped'})
    token=a._worker_scope.set({'session_id':'S1','customer_id':'C1'})
    try:
        worker=a.build_inventory_agent()
        result=function(worker,'list_customer_orders')('C1')
        assert [o['order_id'] for o in result['orders']]==['O1']
        assert function(worker,'check_order_status')('C1','OTHER')=={'error':'order_not_found'}
        with pytest.raises(PermissionError):
            function(worker,'get_customer_tier')('C2')
    finally:
        a._worker_scope.reset(token)


def test_required_temperatures_and_retriever_instances(db,monkeypatch):
    built={}
    original=a._new_agent
    def capture(name,*args,**kwargs):
        agent=original(name,*args,**kwargs)
        built[name]=agent
        return agent
    monkeypatch.setattr(a,'_new_agent',capture)
    a.build_agent_graph()
    expected={'InventoryAgent':0.1,'RefundAgent':0.1,'PolicyAgent':0.2,
        'CommunicationAgent':0.3,'OrchestratorAgent':0.0,
        'ReturnsPolicyRetrieverAgent':0.0,'ShippingPolicyRetrieverAgent':0.0,
        'WarrantyPolicyRetrieverAgent':0.0}
    assert {name:agent.model.config['temperature'] for name,agent in built.items()} == expected
    assert all(len(built[name].tool_registry.registry)==1 for name in expected if 'Retriever' in name)
