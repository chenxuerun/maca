import torch

from util.other_util import TOTAL_UNIT_NUM

def train_ganlu(agent):
    if agent.record == False: return

    agent.recorder.post_process_record()
    agent.commander.train()

    for plane_id in range(1, TOTAL_UNIT_NUM):
        s, a, r, s_prime, is_last = agent.recorder.record_to_dqn_training_data(plane_id)
        s = torch.Tensor(s).cuda()
        a = torch.Tensor(a).cuda()
        r = torch.Tensor(r).cuda()
        s_prime = torch.Tensor(s_prime).cuda()
        is_last = torch.Tensor(is_last).cuda()
        
        if plane_id in [1, 2]:
            agent.commander.dqnd.learn(s, a, r, s_prime, is_last)
        else:
            agent.commander.dqnf.learn(s, a, r, s_prime, is_last)

    agent.commander.save_model()