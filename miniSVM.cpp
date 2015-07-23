#include<vector>
#include<algorithm>
#include<functional>
#include<iostream>
#include<unistd.h>
#include<cstdio>
#include<cstdlib>
#include<fstream>
#include<strstream>
#include<iomanip>
#include<cmath>
#include<ctime>
#include<cstdio>
#include<cstdarg>
#include "rand.h"
using namespace std;


//全局变量

int N = 0; //N个训练点
int d = -1;
double C = 0.05; //惩罚因子
double tolerance = 0.001;
double eps = 0.001;
double two_sigma_squared = 2;

vector<double> alph;//拉格朗日算子
double b;           //阈值
vector<double> w;   //权值，仅对线性核

vector<double> error_cache;

struct sparse_binary_vector{
    vector<int> id;
};

struct sparse_vector{
    vector<int> id;
    vector<double> val;
};

typedef vector<double> dense_vector;

bool is_sparse_data = false;
bool is_binary = false;

//只是用下面其中的一个
vector<sparse_binary_vector> sparse_binary_points;
vector<sparse_vector> sparse_points;
vector<dense_vector> dense_points;

vector<int> target;//训练集数据的类标
bool is_test_only = false;
bool is_linear_kernel = false;

int first_test_i = 0;

int end_support_i = -1;

double delta_b;



vector<double> precomputed_self_dot_product;

double (*learned_func)(int) = NULL;
double (*dot_product_func)(int,int)=NULL;
double (*kernel_func)(int,int) = NULL;

//end global variables
//begin functions


/****************输出函数*******************/
static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//以下以learned_func开头的是分类目标函数

/**
线性稀疏二分类问题
*/
double learned_func_linear_sparse_binary(int k){
    double s = 0.;

    for(int i=0; i<sparse_binary_points[k].id.size(); i++)
        s += w[sparse_binary_points[k].id[i]];
    s -= b;

    return s;
}

//线性系数非二分类问题
double learned_func_linear_sparse_nobinary(int k){
    double s = 0.;

    for(int i=0; i<sparse_points[k].id.size(); i++)
    {
        int j = sparse_points[k].id[i];
        double v = sparse_points[k].val[i];
        s += w[j]*v;
    }

    s -= b;
    return s;
}

//线性稠密非二分类问题
double learned_func_linear_dense(int k){
    double s = 0.;

    for(int i=0; i<d; i++)
        s += w[i]*dense_points[k][i];
    s -= b;
    return s;
}

//非线性分类问题
double learned_func_nolinear(int k){
    double s = 0.;

    for(int i=0; i<end_support_i; i++)
        if(alph[i] > 0)
            s += alph[i]*target[i]*kernel_func(i,k);
    s -= b;
    return s;
}

//以下以dot_product 开头的是点积函数
//kernel functions and dot product functions
double dot_product_sparse_binary(int i1,int i2){
    int p1=0,p2=0,dot=0;
    int num1 = sparse_binary_points[i1].id.size();
    int num2 = sparse_binary_points[i2].id.size();

    while(p1 < num1 && p2 < num2)
    {
        int a1 = sparse_binary_points[i1].id[p1];
        int a2 = sparse_binary_points[i2].id[p2];

        if(a1 == a2)
        {
            dot++;
            p1++;
            p2++;
        }
        else if(a1 < a2)
            p1++;
        else
            p2++;
    }

    return double(dot);
}


double dot_product_sparse_nobinary(int i1,int i2){
    int p1=0,p2=0;
    double dot=0.;
    int num1 = sparse_points[i1].id.size();
    int num2 = sparse_points[i2].id.size();

    while(p1 < num1 && p2 < num2)
    {
        int a1 = sparse_points[i1].id[p1];
        int a2 = sparse_points[i2].id[p2];

        if(a1 == a2)
        {
            dot += sparse_points[i1].val[p1] *sparse_points[i2].val[p2];
            p1++;
            p2++;
        }
        else if(a1 < a2)
            p1++;
        else
            p2++;
    }
    return dot;
}

double dot_product_dense(int i1,int i2){
    double dot = 0.;
    for(int i=0; i<d; i++)
        dot += dense_points[i1][i] * dense_points[i2][i];
    return dot;
}


/**
快速计算
||x1-x2||^2 = x1^T*x1 + x2^T*x2 - 2x1^T*x2;
*/
double rbf_kernel(int i1, int i2){
    double s = dot_product_func(i1,i2);
    s *= (-2);
    s += precomputed_self_dot_product[i1] + precomputed_self_dot_product[i2];

    return exp(-s/two_sigma_squared);
}


double error_rate()
{
    int n_total = 0;
    int n_error = 0;
    for(int i = first_test_i; i<N; i++)
    {
        if((learned_func(i) > 0) != (target[i] > 0))
            n_error++;
        n_total++;
    }
    return double(n_error) / double(n_total);
}

/**********************************/
/*
优化两个变量alph1, alph2
*/
int takeStep(int i1,int i2)
{
    int y1,y2,s;
    double alph1,alph2;//旧的的alph值
    double a1,a2;//新的alph值
    double E1,E2,L,H,k11,k22,k12,eta,Lobj,Hobj;

    if(i1 == i2)
        return 0;
    //计算alph1,y1,E1,alph2,y2,E2的值
    {
        alph1 = alph[i1];
        y1 = target[i1];
        if(alph1 > 0 && alph1 < C)
            E1 = error_cache[i1];
        else
            E1 = learned_func(i1) - y1;

        alph2 = alph[i2];
        y2 = target[i2];
        if(alph2 > 0 && alph2 < C)
            E2 = error_cache[i2];
        else
            E2 = learned_func(i2) - y2;
    }

    s = y1 * y2;
    //计算L,H的值
    /**
    从a1，a2的二维空间来确定值
    if y1 == y2
        a1 + a2 = r (gamma)
        L = max(0,aj + ai - C) = max(0,r - C)
        H = min(C,aj + ai) = min(C , r)
    else
        a1 - a2 = r (gamma)
        L = max(0,aj - ai) = max(0, -r)
        H = min(C,C + aj - ai) = min(C, C - r)

    */
    {
        if(y1 == y2)
        {
            double gamma = alph1 + alph2;
            if(gamma > C)
            {
                L = gamma - C;
                H = C;
            }
            else{
                L = 0;
                H = gamma;
            }
        }
        else{
            double gamma = alph1 - alph2;
            if(gamma > 0){
                L = 0;
                H = C - gamma;
            }
            else{
                L = -gamma;
                H = C;
            }
        }
    }

    if(L == H)
        return 0;

    //计算eta的值
    {
        k11 = kernel_func(i1,i1);
        k12 = kernel_func(i1,i2);
        k22 = kernel_func(i2,i2);
        eta = 2*k12 - k11 - k22;
    }

    //计算a2 的值，如果eta不为0，说明可导，可以使用公式
    if(eta < 0){
        a2 = alph2 + y2 * (E2 - E1)/eta;
        if(a2 < L)
            a2 = L;
        else if(a2 > H)
            a2 = H;
    }
    //eta = 0，则计算两个端点
    else
    {
        {
            double c1 = eta / 2;
            double c2 = y2 *(E1 - E2) - eta * alph2;
            Lobj = c1 * L * L + c2 * L;
            Hobj = c1 * H * H + c2 * H;
        }
        if(Lobj > Hobj +eps)
            a2 = L;
        else if(Lobj < Hobj - eps)
            a2 = H;
        else
            a2 = alph2;
    }

    if(fabs(a2 - alph2) < eps*(a2+alph2 + eps))
        return 0;

    a1 = alph1 - s * (a2 - alph2);

    if(a1 < 0)
    {
        a2 += s * a1;
        a1 = 0;
    }
    else if(a1 > C)
    {
        double t = a1 -C;
        a2 += s * t;
        a1 = C;
    }

    //更新拉格朗日乘子
    {
        double b1,b2,bnew;

        if(a1 > 0 && a1 < C)
            bnew = b + E1 + y1 * (a1- alph1)*k11 + y2 *(a2 - alph2)*k12;
        else{
            if ( a2 > 0 && a2 < C)
                bnew = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
            else{
                b1 = b + E1 + y1 * (a1- alph1)*k11 + y2 *(a2 - alph2)*k12;
                b2 = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
                bnew = (b1 + b2)/2;
            }
        }

        delta_b  = bnew - b;

        b = bnew;
    }

    //更新权重值，如果是线性核
    {
        if(is_linear_kernel)
        {
            double t1 = y1 * (a1 - alph1);
            double t2 = y2 * (a2 - alph2);

            if(is_sparse_data && is_binary)
            {
                int p1,num1,p2,num2;

                num1 = sparse_binary_points[i1].id.size();
                for(p1=0; p1<num1; p1++)
                {
                    w[sparse_binary_points[i1].id[p1]] += t1;
                }
                num2 = sparse_binary_points[i2].id.size();
                for(p2=0; p2 < num2; p2++)
                    w[sparse_binary_points[i2].id[p2]] += t2;

            }
            else if(is_sparse_data && !is_binary)
            {
                int p1,num1,p2,num2;
                num1 = sparse_points[i1].id.size();
                for(p1=0; p1<num1; p1++)
                    w[sparse_points[i1].id[p1]] += sparse_points[i1].val[p1] * t1;

                num2 = sparse_points[i2].id.size();
                for(p2=0; p2<num2; p2++)
                    w[sparse_points[i2].id[p2]] += sparse_points[i2].val[p2] * t2;

            }
            else{
                for(int i=0; i<d; i++)
                    w[i] += dense_points[i1][i] * t1 + dense_points[i2][i] * t2;
            }
        }
    }
    //更新拉格朗日error_cache 缓存
    {
        double t1 = y1 * (a1 - alph1);
        double t2 = y2 * (a2 - alph2);

        for(int i=0; i<end_support_i; i++)
            if(alph[i] > 0 && alph[i] < C)
                error_cache[i] += t1 * kernel_func(i1,i) + t2 * kernel_func(i2,i) - delta_b;
        
        //i1 i2这两个点已经分类正确，分类错误为0
        error_cache[i1] = 0.;
        error_cache[i2] = 0.;
    }

    alph[i1] = a1;
    alph[i2] = a2;

    return 1;
}

/*
检验第i1个训练样例是否违反KKT条件，违反返回1，否则返回0
*/
int examineExample(int i1)
{
    double y1,alph1,E1,r1;

    y1 = target[i1];
    alph1 = alph[i1];

    if(alph1 > 0 && alph1 < C)
        E1 = error_cache[i1];
    else
        E1 = learned_func(i1)-y1;

    r1 = y1*E1;

    //寻找第一个违反KKT条件的i1
    if((r1 < -tolerance && alph1 < C)
       ||(r1 > tolerance && alph1 > 0))
    {//有3种方式来寻找i2
        //1. 从所有支持向量中找i2 使得最大化|E1-E2|
        {
            int k , i2;
            double tmax = 0;
            i2 = -1;
            for(k = 0; k < end_support_i; k++)
            {
                if(alph[k] >0 && alph[k] < C)
                {
                    double E2,temp;

                    E2 = error_cache[k];
                    temp = fabs(E1-E2);
                    if(temp > tmax)
                    {
                        tmax = temp;
                        i2 = k;
                    }
                }
            }
            if(i2 >= 0)
            {
                if(takeStep(i1,i2))
                    return 1;
            }
        }
        //2. 从支持向量里随机找一个
        {
            int k,k0;
            int i2;
            k0 = (int)(drand48()*end_support_i);
            for(k=k0; k<end_support_i + k0; k++)
            {
                i2 = k % end_support_i;
                if(alph[i2] > 0 && alph[i2] < C)
                {
                    if(takeStep(i1,i2))
                        return 1;
                }
            }
        }
        //3. 从全部训练数据集里找任意找一个
        {
            int k0,k,i2;

            k0 = int(drand48()*end_support_i);
            for(k=k0; k<end_support_i+k0; k++)
            {
                i2 = k % end_support_i;
                if(takeStep(i1,i2))
                    return 1;
            }
        }
    }

    return 0;
}



int read_data(istream& is)
{
//    info("reading data now ...\n");
    string s;
    int n_lines;
    for(n_lines = 0; getline(is,s,'\n'); n_lines++)
    {
        istrstream line(s.c_str());
        vector<double> v;
        vector<int> indexs;
        int index;
        double val;
        char c;

        int t;
        line >>t;
        target.push_back(t);
        while(line >> index >> c >> val){
            indexs.push_back(index);
            v.push_back(val);
        }

//        int n = v.size();
        if(is_sparse_data && is_binary)
        {
//            info("is sparse data && binary\n");
            sparse_binary_vector x;
            for(int i=0; i<v.size(); i++){
                x.id.push_back(indexs[i]);
                if(d < indexs[i])
                    d = indexs[i];
            }
             sparse_binary_points.push_back(x);

        }else if(is_sparse_data && !is_binary)
        {
//             info("is sparse data && not binary\n");
            sparse_vector x;
            for(int i=0; i<v.size(); i++)
            {
                x.id.push_back(indexs[i]);
                x.val.push_back(v[i]);
                //获取特征维数
                if(d < indexs[i])
                    d = indexs[i];

            }
            sparse_points.push_back(x);
        }
        else{
            d = v.size();
//            if(v.size() != d)
//            {
//                cerr<<"data file error : line "<< n_lines + 1
//                    <<" has " << v.size() << " attributes ; should be d="<<d
//                    <<endl;
//                    exit(1);
//            }
            dense_points.push_back(v);
        }
    }

//    for(int i=0; i<dense_points.size(); i++)
//    {
//        cout<<target[i]<<" ";
//        for(int j=0; j<dense_points[i].size(); j++)
//            cout<<" "<<dense_points[i][j];
//        cout<<endl;
//    }
//    cout<<d<<endl;
//    for(int i=0; i<sparse_binary_points.size(); i++){
//        for(int j=0; j<sparse_binary_points[i].id.size(); j++)
//            cout<<" "<<sparse_binary_points[i].id[j]<<":1";
//        cout<<endl;
//        break;
//    }

    info("read data over!\n");
    return n_lines;
}


void write_svm(ostream& os)
{
    os <<"d "<<d <<endl;
    os <<"sparse "<<is_sparse_data <<endl;
    os <<"binary "<<is_binary <<endl;
    os <<"linear "<< is_linear_kernel <<endl;
//    os <<"kerne"
    os <<"b "<< b << endl;
    if(is_linear_kernel){
        os <<"w"<<endl;
        for (int i=0; i<d; i++)
            os << w[i] <<endl;
    }
    else{
        os<<"two_sigma_squared "<< two_sigma_squared <<endl;
        int n_support_vectors = 0;
        for(int i=0; i<end_support_i; i++)
            if(alph[i] > 0)
                n_support_vectors++;
        os<<"nSV "<<n_support_vectors<<endl;
        for(int i=0; i<end_support_i; i++)
            if(alph[i] > 0)
                os<<alph[i]<<" ";
        cout<<endl;

        for(int i=0; i<end_support_i; i++)
        {
            if(alph[i] > 0)
            {
                os<<target[i];
                if(is_sparse_data && is_binary)
                {
                    for(int j=0; j<sparse_binary_points[i].id.size(); j++)
                        os<<" "<<sparse_binary_points[i].id[j]<<":1";
                }
                else if(is_sparse_data && !is_binary)
                {
                    for(int j=0; j<sparse_points[i].id.size();j++)
                        os<<" "<<sparse_points[i].id[j]<<":"<<sparse_points[i].val[j];
                }
                else{
                    for(int j=0; j<d; j++)
                        os <<" "<<(j+1)<<":"<< dense_points[i][j];
                }
                os<<endl;
            }
        }
    }

}


int main(int argc , char *argv[] )
{
//    cout<<"Hello World"<<endl;
    //file name
    char *data_file_name = "svm.data";
    char *svm_file_name = "svm.model";
    char *output_filename_name = "svm.output";


    int numChanged;
    int examineAll;

    //读取参数
    {
        extern char *optarg;
        extern int optind;
        int c;
        int errflg = 0;
//        for(int i=0; i<argc; i++)
//            cout<<argv[i]<<" ";
//        cout<<endl;
        //getopt读取参数函数
        while((c = getopt(argc,argv,"n:d:c:t:e:p:f:m:o:r:lsba")) != EOF)
        {
            switch(c)
            {
            case 'n':
                N = atoi(optarg);
//                printf("%s\n",optarg);
//                cout<<"N is "<<N<<endl;
                break;
            case 'd':
                d = atoi(optarg);
                break;
            case 'c':
                C = atof(optarg);
                break;
            case 't':
                tolerance = atof(optarg);
                break;
            case 'e':
                eps = atof(optarg);
                break;
            case 'p':
                two_sigma_squared = atof(optarg);
                break;
            case 'f':
                data_file_name = optarg;
                cout<<"data file name: "<<data_file_name<<endl;
                break;
            case 'm':
                svm_file_name = optarg;
                break;
            case 'o':
                output_filename_name = optarg;
                break;
//            case 'r':
//                srand48(atoi(optarg));
//                break;
            case 'l':
                is_linear_kernel = true;
//                cout<<"is_linear_kernel is true"<<endl;
                break;
            case 's':
                is_sparse_data = true;
                break;
            case 'b':
                is_binary = true;
                break;
            case 'a':
                is_test_only = true;
                break;
            case '?':
                errflg++;
            }
        }

      if(errflg || optind < argc)
        {
            cerr<<"usage: "<< argv[0] <<" "<<
                "-f data_file_name\n\t"\
                "-m svm_file_name\n\t"\
                "-o output_file_name\n\t"\
                "-n N\n\t"
                "-d d\n\t"
                "-c C\n\t"
                "-t tolerance\n\t"
                "-e epsilon\n\t"
                "-p two_sigma_squared\n\t"
                "-l (is_linear_kernel)\n\t"
                "-s (is_sparse_data)\n\t"
                "-b (is_binary)\n\t"
                "-a (is_test_only)\n"
                ;
            exit(2);
        }
    }//end 读取参数

    srand48(time(0));

    //read data
    {
        int n;
        if(is_test_only)
        {
/* TODO (tan#1#): test only code */

        }
        if(N > 0){
            target.reserve(N);
            if(is_sparse_data && is_binary)
                sparse_binary_points.reserve(N);
            else if(is_sparse_data && ! is_binary)
                sparse_points.reserve(N);
            else
                dense_points.reserve(N);
        }

        ifstream data_file(data_file_name);
        n = read_data(data_file);
        if(is_test_only)
            N = first_test_i + n;
        else{
            N = n;
            first_test_i = 0;
            end_support_i = N;
        }
    }

    if(!is_test_only)
    {
        alph.resize(end_support_i,0.);

        b = 0;

        error_cache.resize(N);

        if(is_linear_kernel)
            w.resize(d,0.);
    }

    if( !is_linear_kernel)
        two_sigma_squared = d;

    //根据数据的类型，使用不同的评价函数
    {
        if(is_linear_kernel && is_sparse_data && is_binary)
            learned_func = learned_func_linear_sparse_binary;
        else if(is_linear_kernel && is_sparse_data && !is_binary)
            learned_func = learned_func_linear_sparse_nobinary;
        else if(is_linear_kernel && !is_sparse_data)
            learned_func = learned_func_linear_dense;
        else if(!is_linear_kernel)
            learned_func = learned_func_nolinear;
    }
    //确定点积函数
    {
        if(is_sparse_data && is_binary)
            dot_product_func = dot_product_sparse_binary;
        else if(is_sparse_data && !is_binary)
            dot_product_func = dot_product_sparse_nobinary;
        else if( !is_sparse_data)
            dot_product_func = dot_product_dense;
    }
    //确定核函数
    {
        if(is_linear_kernel)
            kernel_func = dot_product_func;
        if( !is_linear_kernel)
            kernel_func = rbf_kernel;
    }
    //计算自身的点积
    {
        if( !is_linear_kernel)
        {
            precomputed_self_dot_product.resize(N);
            for(int i=0; i<N; i++)
                precomputed_self_dot_product[i] = dot_product_func(i,i);
        }
    }

//是否运行计算过程

#if 1
    if( !is_test_only)
    {
        //寻找第一个第一个a1
        numChanged = 0;
        examineAll = 1;
        //所有的点都已经分类正确
        while(numChanged > 0 || examineAll)
        {
            numChanged = 0;
            if(examineAll){
                for(int k=0; k<N; k++)
                    numChanged += examineExample(k);
            }
            else{
                for(int k=0; k<N; k++)
                    if(alph[k] !=0 && alph[k] != C)
                        numChanged += examineExample(k);
            }
            
            cout<<"examineAll= "<<examineAll<<" numChanged= "<< numChanged<<endl;
            
            //
            if(examineAll == 1)
                examineAll = 0;
            else if(numChanged == 0)
                examineAll = 1;

        {
#if 1
            
    #if 0

           double s = 0.;
           for(int i=0; i<N; i++)
                s +=alph[i];
           double t = 0.;
           for(int i=0; i<N; i++)
                for(int j=0; j<N; j++)
                    t += alph[i]*alph[j]*target[i]*target[j]*kernel_func(i,j);

           cerr<<"Objective function= "<<(s-t/2)<<endl;
           for(int i=0; i<N; i++)
                if(alph[i] < 0)
                    cerr<<"alph["<<i<<"]="<<alph[i]<<" <0"<<endl;

           s = 0.;
           for(int i=0; i<N; i++)
                s += alph[i]*target[i];
           cerr<<"s="<<s<<endl;

    #endif // 1
            cerr<< "error rate: "<< error_rate()<<endl;
            int non_bound_support = 0;
            int bound_support = 0;
            for(int i=0; i<N; i++)
                if(alph[i] > 0){
                    if(alph[i] < C)
                        non_bound_support++;
                    else
                        bound_support++;
                }
            cerr<<"non_bound="<<non_bound_support<<endl;
            cerr<<"bound_support="<<bound_support<<endl;
            cout<<endl;
#endif // 1
        }



        }//end while

    {
        if( !is_test_only && svm_file_name != NULL)
        {
            ofstream svm_file(svm_file_name);
            write_svm(svm_file);
        }
    }
        cerr<<"threshold b= "<< b<<endl;
    }

    cerr<< "Accuracy rate: "<< 1 - error_rate()<<endl;
//    cerr<< "Error rate: "<< error_rate()<<endl;

#endif // 0


    return 0;

}





