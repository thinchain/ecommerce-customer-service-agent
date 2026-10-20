"""Run BrowseUser and ChatUser SEPARATELY; fast HTTP throughput is not LLM throughput."""
from locust import HttpUser, between, task


class BrowseUser(HttpUser):
    wait_time = between(.2,1)
    @task
    def overview(self):
        self.client.get('/api/commerce/overview',name='catalog_overview')


class ChatUser(HttpUser):
    wait_time = between(1,3)
    def on_start(self):
        self.sid = self.client.post('/api/sessions').json()['session_id']
    @task
    def ask(self):
        # One session per simulated user, avoiding artificial same-session 409 conflicts.
        with self.client.post('/api/chat',json={'message':'查询 ORD00001 的物流状态','session_id':self.sid},
                              timeout=190,name='chat_order',catch_response=True) as response:
            if response.status_code==429:
                response.failure('admission_rejected_429')
            elif response.status_code==200 and not response.json().get('reply'):
                response.failure('empty_answer')
    def on_stop(self):
        if getattr(self,'sid',None): self.client.delete('/api/sessions/'+self.sid,name='delete_session')
