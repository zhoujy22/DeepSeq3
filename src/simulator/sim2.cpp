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
vector<int> random_pattern_generator(int length) {
    vector<int> res(length);
    for (int i = 0; i < length; i++) {
        res[i] = rand() % 2;
    }
    return res;
}
vector<int> dec_to_bin_array(int num, int N) {
    vector<int> res(N, 0);
    for (int i = N - 1; i >= 0; --i) {
        res[i] = num & 1;
        num >>= 1;
    }
    return res;
}

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

// -------- 单步仿真（给定当前状态 + 当前PI组合） --------
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

    // 初始化 DFF 输出
    int k = 0;
    for (int idx = 0; idx < (int)x_data.size(); idx++) {
        if (x_data[idx].second == gate_to_index.at("DFF")) {
            state[idx] = curr_state_bits[k++];
        }
    }
    // 初始化 PI
    for (int i = 0; i < (int)PI_indexes.size(); i++) {
        state[PI_indexes[i]] = pi_bits[i];
    }

    int tmp_level = 1;
    vector<int> visited = PI_indexes;
    for (auto &x : x_data) {
        if (x.second == gate_to_index.at("DFF")) {
            visited.push_back(x.first);
        }
    }
    int num_visited = visited.size();

    while (num_visited < (int)x_data.size()) {
        int count = 0;
        for (int level = tmp_level; level < (int)level_list.size(); level++) {
            for (int node_idx : level_list[level]) {
                if (find(visited.begin(), visited.end(), node_idx) != visited.end()
                    && x_data[node_idx].second != gate_to_index.at("DFF")) {
                    continue;
                }
                vector<int> inputs;
                bool flag = false;
                for (int pre_idx : fanin_list[node_idx]) {
                    if (state[pre_idx] == -1) { flag = true; break; }
                    inputs.push_back(state[pre_idx]);
                }
                if (flag) continue;

                count++;
                num_visited++;
                if (!inputs.empty()) {
                    int gate_type = x_data[node_idx].second;
                    int res = logic(gate_type, inputs, gate_to_index);
                    state[node_idx] = res;
                    if (gate_type != gate_to_index.at("DFF")) {
                        visited.push_back(node_idx);
                    }
                }
            }
        }
        if (count == 0) {
            tmp_level = 0;
            for (int node_idx : level_list[tmp_level]) {
                if (x_data[node_idx].second == 4) {
                    string node_name = idx2name.at(x_data[node_idx].first);
                    if (!node_name.empty() && node_name[0] == 'P') {
                        node_name = node_name.substr(1);
                    }
                    for (int idx = 0; idx < (int)x_data.size(); idx++) {
                        if (node_name == idx2name.at(x_data[idx].first) &&
                            find(visited.begin(), visited.end(), idx) != visited.end()) {
                            state[node_idx] = state[idx];
                            visited.push_back(node_idx);
                        }
                    }
                }
            }
            num_visited = visited.size();
        }
        tmp_level++;
    }

    // 收集下一状态（DFF的输出）
    vector<int> result;
    for (int idx = 0; idx < (int)x_data.size(); idx++) {
        if (x_data[idx].second == gate_to_index.at("DFF")) {
            result.push_back(state[idx]);
        }
    }
    return bin_array_to_dec(result);
}

// -------- 一步转移矩阵（布尔，PI穷举） --------
vector<vector<double>> simulate_one_step_matrix(
    const vector<pair<int,int>>& x_data,
    const vector<int>& PI_indexes,
    const vector<vector<int>>& level_list,
    const vector<vector<int>>& fanin_list,
    const unordered_map<string,int>& gate_to_index,
    const unordered_map<int,string>& idx2name,
    int number
) {
    int S = 1 << number;
    int M = 1 << PI_indexes.size() > 1024 ? 1024 : (1 << PI_indexes.size());
    vector<vector<double>> mat(S, vector<double>(S, 0));

    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < S; s++) {
        vector<int> curr_bits = dec_to_bin_array(s, number);
        for (int pi_val = 0; pi_val < M; pi_val++) {
            vector<int> pi_bits = random_pattern_generator(PI_indexes.size());
            int ns = step_with_given_inputs(
                x_data, PI_indexes, level_list, fanin_list,
                gate_to_index, idx2name, curr_bits, pi_bits
            );
            mat[s][ns] += 1.0;
        }
        double sum = std::accumulate(mat[s].begin(), mat[s].end(), 0.0);
        if (sum != 0.0) {
            for (auto &val : mat[s]) {
                val /= sum;
            }
        }
    }
    return mat;
}

// -------- 无穷步可达矩阵（传递闭包） --------
vector<vector<double>> reachability_closure(
    const vector<pair<int,int>>& x_data,
    const vector<int>& PI_indexes,
    const vector<vector<int>>& level_list,
    const vector<vector<int>>& fanin_list,
    const unordered_map<string,int>& gate_to_index,
    const unordered_map<int,string>& idx2name,
    int number
) {
    auto adj = simulate_one_step_matrix(
        x_data, PI_indexes, level_list, fanin_list,
        gate_to_index, idx2name, number
    );
    int S = adj.size();
    for (int i = 0; i < S; i++) adj[i][i] = 1;
    for (int k = 0; k < S; k++ ){
        for (int i = 0; i < S; i ++){
            adj[k][i] = adj[k][i] > 0 ? 1.0: 0.0;
        }
    }
    for (int k = 0; k < S; k++) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < S; i++) {
            if (adj[i][k]) {
                #pragma omp simd
                for (int j = 0; j < S; j++) {
                    if (adj[k][j]) adj[i][j] = 1;
                }
            }
        }
    }
    return adj;
}

// -------- Python绑定 --------
PYBIND11_MODULE(fsm_simulator, m) {
    m.doc() = "FSM simulator with exact one-step and infinite-step reachability";

    m.def("simulate_one_step_matrix", &simulate_one_step_matrix,
          py::arg("x_data"), py::arg("PI_indexes"), py::arg("level_list"),
          py::arg("fanin_list"), py::arg("gate_to_index"), py::arg("idx2name"),
          py::arg("number"));

    m.def("reachability_closure", &reachability_closure,
          py::arg("x_data"), py::arg("PI_indexes"), py::arg("level_list"),
          py::arg("fanin_list"), py::arg("gate_to_index"), py::arg("idx2name"),
          py::arg("number"));
}
