#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <set>
#include <utility>
#include <functional>
#include <numeric>
#include <iostream>
#include <cassert>

#define MAX_DISK_NUM (10 + 1)
#define MAX_DISK_SIZE (16384 + 1)
#define MAX_REQUEST_NUM (30000000 + 1)
#define MAX_OBJECT_NUM (100000 + 1)
#define REP_NUM (3)
#define FRE_PER_SLICING (1800)
#define EXTRA_TIME (105)
#define MAX_OBJECT_SIZE (5 + 1)
#define MAX_TAG_NUM (16 + 1)
#define MAX_STAGE (50)

const int READ_ROUND_TIME = 5; //一轮读取的时间

struct _Object {
    //(磁盘编号，磁盘内位置)
    std::pair <char, short> unit_pos[REP_NUM + 1][MAX_OBJECT_SIZE];
    char size;
    char tag;
    //读入的时候注意赋值给char型
};

struct _Request {
    int object_id;
};

_Request requests[MAX_REQUEST_NUM];

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意，objects加了复数
_Object objects[MAX_OBJECT_NUM];

int T, M, N, V, G;

struct Segment_tree_max {
    struct Node {
        int max, pos;
        Node() {}
        Node(int _max, int _pos) : max(_max), pos(_pos) {}
        friend Node operator + (const Node& x, const int& num) {
            return Node(x.max + num, x.pos);
        }

        friend bool operator < (const Node& x, const Node& y) {
            return x.max < y.max;
        }
    };

    Node seg[MAX_DISK_SIZE << 2];
    int add_tag[MAX_DISK_SIZE << 2];

    void build(int o, int l, int r) {
        if (l == r) {
            seg[o].pos = l;
            return ;
        }

        int mid = l + r >> 1;
        build(o << 1, l, mid);
        build(o << 1 | 1, mid + 1, r);
        seg[o] = seg[o << 1];
    }

    inline void apply(int o, int now_ad) 
    {
        add_tag[o] += now_ad;
        seg[o] = seg[o] + now_ad;
    }

    inline void push_down(int o) 
    {
        if (!add_tag[o]) return ;
        apply(o << 1, add_tag[o]);
        apply(o << 1 | 1, add_tag[o]);
        add_tag[o] = 0;
    }

    inline void push_up(int o) 
    {
        seg[o] = std::max(seg[o << 1], seg[o << 1 | 1]);
    }
    
    //返回差值，方便后面维护density
    int modify(int o, int l, int r, int p, int v) 
    {
        if (l == r) {
            int pre = seg[o].max;
            seg[o] = Node(v, l);
            return v - pre;
        }
        
        push_down(o);
        int mid = l + r >> 1, res = 0;
        if (p <= mid)
            res = modify(o << 1, l, mid, p, v);
        else res = modify(o << 1 | 1, mid + 1, r, p, v);

        push_up(o);
        return res;
    }

    void add(int o, int l, int r, int x, int y, int ad) 
    {
        if (x <= l && y >= r) {
            return apply(o, ad);
        }

        int mid = l + r >> 1;
        push_down(o);
        if (x <= mid)
            add(o << 1, l, mid, x, y, ad);
        if (y > mid)
            add(o << 1 | 1, mid + 1, r, x, y, ad);
        push_up(o);
    }

    Node query_max(int o, int l, int r, int x, int y) 
    {
        if (x <= l && y >= r) return seg[o];
        push_down(o);
        int mid = l + r >> 1;
        Node res = Node(-1, -1);

        if (x <= mid)
            res = query_max(o << 1, l, mid, x, y);
        if (y > mid)
            res = std::max(res, query_max(o << 1 | 1, mid + 1, r, x, y));
        return res;    
    }

    int find_max_point() {
        return query_max(1, 1, V, 1, V).pos;
    }

    int find_next(int o, int l, int r, int x, int y, int lim) 
    {
        if (seg[o].max < lim) return -1;
        if (l == r) return l;

        int mid = l + r >> 1;
        int best = -1;
        push_down(o);
        if (x <= mid) 
            best = find_next(o << 1, l, r, x, y, lim);
        if (best == -1 && y > mid) 
            best = find_next(o << 1 | 1, mid + 1, r, x, y, lim);

        return best;
    }

    int find_next(int p, int lim) {
        int nxt = find_next(1, 1, V, p + 1, V, lim);
        if (nxt != -1)
            return nxt;
        return find_next(1, 1, V, 1, p, lim);
    }
};

struct Segment_tree_add {
    int seg[MAX_DISK_SIZE << 2];
    void set_one(int o, int l, int r) 
    {
        if (l == r) return seg[o] = 1, void();
        int mid = l + r >> 1;
        set_one(o << 1, l, mid), set_one(o << 1 | 1, mid + 1, r);
        seg[o] = seg[o << 1] + seg[o << 1 | 1];
    }

    int add_unit(int o, int l, int r, int p) 
    {
        if (seg[o] < p) return -1;
        if (l == r) {
            seg[o] = 0;
            return l;
        }

        int mid = l + r >> 1;
        int res = -1;
        res = add_unit(o << 1, l, mid, p);
        if (res == -1) {
            res = add_unit(o << 1 | 1, mid + 1, r, p - seg[o << 1]);
        }

        seg[o] = seg[o << 1] + seg[o << 1 | 1];
        return res;
    }

    void modify(int o, int l, int r, int p, int v) {
        if (l == r) {
            seg[o] += v;
            return ;
        }

        int mid = l + r >> 1;
        if (p <= v)
            modify(o << 1, l, mid, p, v);
        else modify(o << 1 | 1, mid + 1, r, p, v);
        seg[o] = seg[o << 1] + seg[o << 1 | 1];
    }

    void delete_unit(int o, int l, int r, int p)
    {
        if (l == r) return seg[o] = 1, void();
        int mid = l + r >> 1;
        if (p <= mid) 
            delete_unit(o << 1, l, mid, p);
        else delete_unit(o << 1 | 1, mid + 1, r, p);
        seg[o] = seg[o << 1] + seg[o << 1 | 1];
        return;
    }

    int find_next(int o, int l, int r, int x, int y)
    {
        if (seg[o] == 0) return -1;
        if (l == r) return l;

        int mid = l + r >> 1;
        int res = -1;
        if (x <= mid)
            res = find_next(o << 1, l, mid, x, y);
        if (y > mid && res == -1)
            res = find_next(o << 1 | 1, mid + 1, r, x, y);
        return res;
    }

    inline int query_rest_unit() {
        return seg[1];
    }
};

struct DISK {
    Segment_tree_add empty_pos; //维护空位置
    Segment_tree_max request_num; //维护每个点的request数量
    Segment_tree_max max_density; //用于获取每个段的request总和
    int pointer; //这个磁盘指针的位置
    int last_read_cnt = 0; //上一次操作往前连续读取的次数
    int last_read_cost = -1; //-1: 上一次不为读取操作 否则为上一次读取操作的花费
    int rest_token;
    int tag_order[MAX_TAG_NUM]; //每个标签在这个磁盘的固定顺序
    int tag_distribution[MAX_TAG_NUM];
    std::pair <int, int> unit_object[MAX_DISK_SIZE];
    int test_density_len = 300;
};

DISK disk[MAX_DISK_NUM];

struct Predict {
    int add_object;
    int delete_object;
    int read_object;
};

Predict Info[MAX_TAG_NUM][MAX_STAGE];

/*存储每个对象的unit没有解决的request*/
std::queue<int> unsolve_request[MAX_OBJECT_NUM][MAX_OBJECT_SIZE];
char request_rest_unit[MAX_REQUEST_NUM];
std::vector <int> solved_request;

/*从x到y的距离*/
inline int get_dist(int x, int y) 
{
    return x <= y? y - x: V - x + y;
}

/*预处理操作*/
void init() 
{
    /*
    read_cost[0] = read_cost[1] = 64;
    for (int i = 2; i < 9; ++i) {
        read_cost[i] = read
    }
    for (int i = 8; i >= )
    */
}

void timestamp_action()
{
    int timestamp;
    scanf("%*s%d", &timestamp);
    printf("TIMESTAMP %d\n", timestamp);

    fflush(stdout);
}

/*未完成的请求*/
std::vector <int> abort_request;

//维护density
inline void modify_max_density(int disk_id, int pos, int delta_request) {
    int pre_pos = std::max(1, pos - disk[disk_id].test_density_len + 1);
    disk[disk_id].max_density.add(1, 1, V, pre_pos, pos, delta_request);
    
    if (pre_pos != pos - disk[disk_id].test_density_len + 1) {
        int rest_num = disk[disk_id].test_density_len - pos;
        disk[disk_id].max_density.add(1, 1, V, V - rest_num + 1, V, delta_request);
    }
}

inline void modify_unit_request(int disk_id, int pos, int value) {
    int delta_request = disk[disk_id].request_num.modify(1, 1, V, pos, value);
    modify_max_density(disk_id, pos, delta_request);
}

inline void add_unit_request(int disk_id, int pos) {
    disk[disk_id].request_num.add(1, 1, V, pos, pos, 1);
    modify_max_density(disk_id, pos, 1);
}

inline void do_object_delete(int object_id) 
{
    //delete pos
    for (int i = 1; i <= 3; ++i) {
        for (int j = 1; j <= objects[object_id].size; ++j) {
            auto [disk_id, pos] = objects[object_id].unit_pos[i][j];

            //维护空位置
            disk[disk_id].empty_pos.delete_unit(1, 1, V, pos);
            //清除request
            modify_unit_request(disk_id, pos, 0);
            disk[disk_id].unit_object[pos] = {0, 0};
        }
    }

    //find abort req
    for (int i = 1; i <= objects[object_id].size; ++i) {
        while (!unsolve_request[object_id][i].empty()) {
            int now_request = unsolve_request[object_id][i].front();

            if (request_rest_unit[now_request] > 0) {
                abort_request.push_back(now_request);
                request_rest_unit[now_request] = -1;
            }
        }
    }
}

void delete_action()
{
    int n_delete;
    int abort_num = 0;
    static int _id[MAX_OBJECT_NUM];

    scanf("%d", &n_delete);
    for (int i = 1; i <= n_delete; i++) {
        scanf("%d", &_id[i]);
        do_object_delete(_id[i]);
    }

    printf("%ld\n", abort_request.size());
    for (int req_id : abort_request) {
        printf("%d\n", req_id);
    }

    fflush(stdout);
}

inline void write_unit(int object_id, int disk_id, int unit_id, int write_pos, int repeat_id) {
    disk[disk_id].empty_pos.add_unit(1, 1, V, 1);
    // disk[disk_id].empty_pos.modify(1, 1, V, write_pos, -1);
    disk[disk_id].unit_object[write_pos] = {object_id, unit_id};
    objects[object_id].unit_pos[repeat_id][unit_id] = {disk_id, write_pos};
}

//no
void write_action()
{
    int n_write;
    scanf("%d", &n_write);
    std::cerr << "n_write: " << n_write << std::endl;
    for (int i = 1; i <= n_write; ++i) {
        std::cerr << "write : "  << i << std::endl;
        int id, size, tag;
        scanf("%d %d %d", &id, &size, &tag);
        objects[id].size = size;
        objects[id].tag = tag;

        std::vector <int> pos(N);
        std::iota(pos.begin(), pos.end(), 1);
        std::random_shuffle(pos.begin(), pos.end());
        int now = 0;
        printf("%d\n", id);
        std::cerr << "object_id : " << id << std::endl;
        for (int j = 1; j <= 3; ++j) {
            int disk_id = pos[now];
            while (disk[disk_id].empty_pos.query_rest_unit() < size) {
                disk_id = pos[++now];
                assert(now < N);
            }
            
            printf("%d ", disk_id);
            std::cerr << "disk_id : " << disk_id << " ";
            for (int k = 1, pre = 0; k <= size; ++k) {
                int nxt = disk[disk_id].empty_pos.find_next(1, 1, V, 1, V);
                // int nxt = disk[disk_id].empty_pos.add_unit(1, 1, V, 1);
                // assert(disk[disk_id].empty_pos.find_next(1, 1, V, 1, V) == nxt + 1);
                write_unit(id, disk_id, k, nxt, j);
                pre = nxt;
                printf("%d ", nxt);
                std::cerr << nxt << " ";
            }

            std::cerr << std::endl;
            printf("\n");
            disk_id = pos[++now];
        }
    }
    fflush(stdout);
}

//use
inline void read_unit(int id, int unit_id) 
{
    while (!unsolve_request[id][unit_id].empty()) {
        int now_request = unsolve_request[id][unit_id].front();
        --request_rest_unit[now_request];
        if (!request_rest_unit[now_request]) {
            solved_request.push_back(now_request);
        }

        unsolve_request[id][unit_id].pop();
    }
}

//use
inline void update_unsolved_request(int request_id, int object_id) 
{
    request_rest_unit[request_id] = objects[object_id].size;

    for (int j = 1; j <= objects[object_id].size; ++j) {
        unsolve_request[object_id][j].push(request_id);       

        for (int i = 1; i <= REP_NUM; ++i) {
            auto [disk_id, unit_id] = objects[object_id].unit_pos[i][j];
            add_unit_request(disk_id, unit_id);
        }
    }
}

//指针进行一次jump
void do_pointer_jump(DISK cur_disk, int destination) 
{
    printf("j %d\n", destination);
    cur_disk.pointer = destination;
    cur_disk.rest_token = 0;
    cur_disk.last_read_cnt = 0;
    cur_disk.last_read_cost = -1;
}

/*
指针进行一次pass
1 success
0 fail
*/
int do_pointer_pass(DISK &cur_disk) 
{
    if (!cur_disk.rest_token)
        return 0;
    printf("p");
    cur_disk.pointer = (cur_disk.pointer + 1) % V + 1;
    cur_disk.rest_token--;
    cur_disk.last_read_cnt = 0;
    cur_disk.last_read_cost = -1;
    return 1;
}
/*
指针进行一次read
1 success
0 fail
*/
int do_pointer_read(DISK &cur_disk) 
{
    int read_cost = cur_disk.last_read_cnt?
        std::max(16, (cur_disk.last_read_cost * 4 + 5 - 1) / 5) : 64;
    if (cur_disk.rest_token < read_cost)
        return 0;
    printf("r");
    
    //清除request
    int& pos = cur_disk.pointer;
    auto [object_id, unit_id] = cur_disk.unit_object[pos];
    if (object_id != 0) {
        for (int i = 1; i <= 3; ++i) {
            auto [disk_id, unit_pos] = objects[object_id].unit_pos[i][unit_id];
            modify_unit_request(disk_id, unit_pos, 0);
        }
    }

    pos = (pos + 1) % V + 1;
    ++cur_disk.last_read_cnt;
    cur_disk.last_read_cost = read_cost;
    return 1;
}

/*
1 选择pass
0 选择read
*/
bool chosse_pass(DISK &cur_disk, int destination)
{
    const int DIST_MIN_PASS = 8;
    int dist = get_dist(cur_disk.pointer, destination);
    if (dist < DIST_MIN_PASS)
        return 0;
    return 1;
}

void read_without_jump(DISK &cur_disk)
{
    while (cur_disk.rest_token > 0) {
        int nxt_p = cur_disk.request_num.find_next(cur_disk.pointer, 1);
        if (nxt_p == -1)
            break;
        if (chosse_pass(cur_disk, nxt_p)) {
            while (cur_disk.rest_token > 0 && cur_disk.pointer < nxt_p)
                do_pointer_pass(cur_disk);
            do_pointer_read(cur_disk);    
        }
        else {
            while (cur_disk.pointer <= nxt_p)
                if (!do_pointer_read(cur_disk))
                    break;
        }
    }
    printf("#\n");
    //int p = disk[cur_disk].request_num.find_next()
}

void read_action(int time)
{
    for (int cur_disk_id = 1; cur_disk_id <= N; ++cur_disk_id)
        disk[cur_disk_id].rest_token = G;

    int n_read;
    int request_id, object_id;
    scanf("%d", &n_read);
    for (int i = 1; i <= n_read; i++) {
        scanf("%d%d", &request_id, &object_id);
        std::cerr << "request_id : " << request_id << " " << "ob : " << object_id << std::endl;
        requests[request_id].object_id = object_id;
        update_unsolved_request(request_id, object_id);
    }

    std::cerr << "in read_action: finish read" << std::endl;
    std::cerr << "in read_action: start move pointer" << std::endl;

    //磁头移动操作
    const int DIST_NOT_JUMP = G;
    for (int cur_disk_id = 1; cur_disk_id <= N; ++cur_disk_id) {
        std::cerr << "cur_disk_id: " << cur_disk_id << std::endl;
        DISK &cur_disk = disk[cur_disk_id];
        if (time % READ_ROUND_TIME == 1) {
            int p = cur_disk.max_density.find_max_point();
            std::cerr << "max_point: " << p << std::endl;
            if (p == -1 || get_dist(cur_disk.pointer, p) <= G) { //如果距离足够近
                read_without_jump(cur_disk);
            }
            else
                do_pointer_jump(cur_disk, p);
        }
        else
            read_without_jump(cur_disk);
    }
    std::cerr << "in read_action: end move pointer" << std::endl;

    //solved request
    printf("%ld\n", solved_request.size());
    for (int request_id : solved_request) {
        printf("%d\n", request_id);
    }

    solved_request.clear();
    fflush(stdout);
}

int main()
{
    std::cerr << "start input global information" << std::endl;
    scanf("%d%d%d%d%d", &T, &M, &N, &V, &G);
    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= (T - 1) / FRE_PER_SLICING + 1; j++) {
            scanf("%*d"); 
        }
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= (T - 1) / FRE_PER_SLICING + 1; j++) {
            scanf("%*d");
        }
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= (T - 1) / FRE_PER_SLICING + 1; j++) {
            scanf("%*d");
        }
    }
    std::cerr << "end input global information" << std::endl;
    init();
    printf("OK\n");
    fflush(stdout);

    for (int i = 1; i <= N; i++) {
        disk[i].pointer = 1;
        disk[i].empty_pos.set_one(1, 1, V);
        disk[i].request_num.build(1, 1, V);
        disk[i].max_density.build(1, 1, V);
    }

    for (int t = 1; t <= T + EXTRA_TIME; t++) {
        std::cerr << "start time " << t << std::endl;
        std::cerr << "start timestamp_action" <<std::endl;
        timestamp_action();
        std::cerr << "end timestamp_action" <<std::endl;
        std::cerr << "start delete_action" <<std::endl;
        delete_action();
        std::cerr << "end delete_action" <<std::endl;
        std::cerr << "start write_action" <<std::endl;
        write_action();
        std::cerr << "end write_action" <<std::endl;
        std::cerr << "start read_action" <<std::endl;
        read_action(t);
        std::cerr << "end read_action" <<std::endl;
        std::cerr << "end time " << t << std::endl;
    }

    return 0;
}