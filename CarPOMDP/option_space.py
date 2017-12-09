import numpy as np

class OptionSpace(object):
    def __init__(self, actions_per_state, max_elements):
        actions_per_state = np.array(actions_per_state)
        self.actions_per_state = actions_per_state
        num_states = len(actions_per_state)
        num_actions = sum(actions_per_state)

        # generate mapping (s,a) pair to value
        self.SA_value = dict()
        self.SA_value_inv = dict()
        idx = 0
        for state in range(num_states):
            for action in range(actions_per_state[state]):
                self.SA_value[(state,action)] = 2 ** idx
                self.SA_value_inv[2 ** idx] = (state,action)
                idx += 1

        # generate the list of valid options
        self.valid_options = list()
        print("Constructing option space this may take a while -- not very efficient")
        policies = np.array(np.unravel_index(
                   range(np.prod(actions_per_state+1, dtype=np.int64)),actions_per_state+1))
        num_terminal = np.sum(policies == actions_per_state.reshape(num_states,1),0)
        below_thresh = ((num_states-num_terminal) <= max_elements)
        policies = policies[:,below_thresh]
        for policy in np.transpose(policies):
            pol = list()
            # convert to (s,a) pair representation
            for state, action in enumerate(policy.tolist()):
                if action != actions_per_state[state]:
                    pol.append((state,action))

            policy_idx = self.encode_policy(pol)
            self.valid_options.append(policy_idx)


    def encode_policy(self, policy):
        idx = 0
        for pair in policy:
            idx += self.SA_value[pair]
        return idx

    def decode_policy(self, idx):
        policy = list()
        policy_states = [True if c == '1' else False
                                for c in reversed(np.binary_repr(idx,9))]
        for idx, value in enumerate(policy_states):
            if value:
                policy.append(self.SA_value_inv[2**idx])

        return policy

    def get_action(self, state, option):
        local_idx = self.idx_global_to_local(option)
        policy = self.decode_policy(local_idx)
        
        return self._get_action(state, policy)

    def _get_action(self, state, policy):
        action = None
        for s, a in policy:
            if s == state:
                return a
        return None #terminal action
                
    def o_new(self, state, option):
        local_idx = self.idx_global_to_local(option)
        policy = self.decode_policy(local_idx)
        
        action = self._get_action(state,policy)
        if action is None:
            return None
        else:
            policy.remove( (state,action) )
            local_idx = self.encode_policy(policy)
            return self.idx_local_to_global(local_idx)

    @property
    def num_options(self):
        return len(self.valid_options)

    def idx_local_to_global(self, idx):
        return self.valid_options.index(idx)

    def idx_global_to_local(self, idx):
        return self.valid_options[idx]

    def options_in_state(self, s):
        valid = list()
        for global_idx, local_idx in enumerate(self.valid_options):
            policy = self.decode_policy(local_idx)
            if self._get_action(s,policy) is not None:
                valid.append(global_idx)
        return valid
