#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

// -------- 工具函数 --------
int bin_array_to_dec(const vector<int>& bin_list) {
    int result = 0;
    for (int x : bin_list) result = (result << 1) | x;
    return result;
}
vector<int> dec_to_bin_array(int num, int N) {
    vector<int> res(N, 0);
    for (int i = N - 1; i >= 0; --i) {
        res[i] = num & 1;
        num >>= 1;
    }
    return res;
}

// 随机输入模式生成器（如果要穷举输入，可以换成 dec_to_bin_array）
vector<int> random_pattern_generator(int length) {
    vector<int> res(length);
    for (int i = 0; i < length; i++) res[i] = rand() % 2;
    return res;
}

// -------- 逻辑函数 --------
int logic(int gate_type, const vector<int>& inputs,
          const unordered_map<string,int>& gate_to_index) {
    if (gate_type == gate_to_index.at("AND"))
        return all_of(inputs.begin(), inputs.end(), [](int x){ return x==1; }) ? 1 : 0;
    else if (gate_type == gate_to_index.at("OR"))
        return any_of(inputs.begin(), inputs.end(), [](int x){ return x==1; }) ? 1 : 0;
    else if (gate_type == gate_to_index.at("NOT"))
        return inputs[0] == 0 ? 1 : 0;
    else if (gate_type == gate_to_index.at("DFF"))
        return inputs[0];
    return 0;
}

int step_with_given_inputs(
    const vector<pair<int,int>>& x_data,
    const vector<int>& PI_indexes,
    const vector<vector<int>>& level_list,
    const vector<vector<int>>& fanin_list,
    const unordered_map<string,int>& gate_to_index,
    const unordered_map<int,string>& idx2name,
    const vector<int>& curr_state_bits,   // 当前状态 (DFF输出Q)
    const vector<int>& pi_bits            // 当前PI输入
) {
    vector<int> state(x_data.size(), -1);

    int k = 0;
    for (int idx = 0; idx < (int)x_data.size(); idx++) {
        if (x_data[idx].second == gate_to_index.at("DFF")) {
            state[idx] = curr_state_bits[k++];
        }
    }
    for (int i = 0; i < (int)PI_indexes.size(); i++) {
        state[PI_indexes[i]] = pi_bits[i];
    }

    for (int level = 1; level < (int)level_list.size(); level++) {
        for (int node_idx : level_list[level]) {
            vector<int> inputs;
            bool ready = true;
            for (int pre_idx : fanin_list[node_idx]) {
                if (state[pre_idx] == -1) { ready = false; break; }
                inputs.push_back(state[pre_idx]);
            }
            if (!ready) continue;
            int gate_type = x_data[node_idx].second;
            state[node_idx] = logic(gate_type, inputs, gate_to_index);
        }
    }

    vector<int> result;
    for (int idx = 0; idx < (int)x_data.size(); idx++) {
        if (x_data[idx].second == gate_to_index.at("DFF")) {
            result.push_back(state[idx]);
        }
    }
    return bin_array_to_dec(result);
}
// -------- 修改后的单步转移（只计算第一行） --------
vector<double> simulate_first_row(
    const vector<pair<int, int>>& x_data,
    const vector<int>& PI_indexes,
    const vector<vector<int>>& level_list,
    const vector<vector<int>>& fanin_list,
    const unordered_map<string, int>& gate_to_index,
    const unordered_map<int, string>& idx2name,
    int number
) {
    int S = 1 << number;
    int M = 1 << PI_indexes.size() > 1024 ? 1024 : (1 << PI_indexes.size());
    vector<double> mat_row(S, 0);

    // 只需要计算 s=0 这一行
    vector<int> curr_bits = dec_to_bin_array(0, number);
    for (int pi_val = 0; pi_val < M; pi_val++) {
        vector<int> pi_bits = random_pattern_generator(PI_indexes.size());
        int ns = step_with_given_inputs(
            x_data, PI_indexes, level_list, fanin_list,
            gate_to_index, idx2name, curr_bits, pi_bits
        );
        mat_row[ns] += 1.0;
    }

    double sum = std::accumulate(mat_row.begin(), mat_row.end(), 0.0);
    if (sum != 0.0) {
        for (auto& val : mat_row) {
            val /= sum;
        }
    }

    return mat_row;
}
vector<int> reachability_row(
    const vector<pair<int,int>>& x_data,
    const vector<int>& PI_indexes,
    const vector<vector<int>>& level_list,
    const vector<vector<int>>& fanin_list,
    const unordered_map<string,int>& gate_to_index,
    const unordered_map<int,string>& idx2name,
    int number,
    int start_state = 0   // 指定起点
) {
    int S = 1 << number;  // 状态总数
    vector<int> reachable(S, 0);
    queue<int> q;

    reachable[start_state] = 1;
    q.push(start_state);

    int max_patterns = min(1 << PI_indexes.size(), 1024); 

    while (!q.empty()) {
        int curr = q.front(); q.pop();
        vector<int> curr_bits = dec_to_bin_array(curr, number);

        for (int pi_val = 0; pi_val < max_patterns; pi_val++) {
            vector<int> pi_bits;
            if ((1 << PI_indexes.size()) <= 1024)
                pi_bits = dec_to_bin_array(pi_val, PI_indexes.size());
            else
                pi_bits = random_pattern_generator(PI_indexes.size());

            int ns = step_with_given_inputs(
                x_data, PI_indexes, level_list, fanin_list,
                gate_to_index, idx2name, curr_bits, pi_bits
            );
            if (!reachable[ns]) {
                reachable[ns] = 1;
                q.push(ns);
            }
        }
    }
    return reachable;
}

// -------- Python绑定 --------
PYBIND11_MODULE(fsm_simulator, m) {
    m.doc() = "FSM simulator with BFS reachability (single row)";
    m.def("simulate_one_step_matrix", &simulate_first_row,
          py::arg("x_data"), py::arg("PI_indexes"), py::arg("level_list"),
          py::arg("fanin_list"), py::arg("gate_to_index"), py::arg("idx2name"),
          py::arg("number"));
    m.def("reachability_row", &reachability_row,
          py::arg("x_data"), py::arg("PI_indexes"), py::arg("level_list"),
          py::arg("fanin_list"), py::arg("gate_to_index"), py::arg("idx2name"),
          py::arg("number"), py::arg("start_state") = 0);
}
