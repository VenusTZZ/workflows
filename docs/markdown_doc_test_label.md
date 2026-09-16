# markdown 测试文档标签

## 设计实现基础
整个设计思路是凭借这个markdown的代码块的 \``` 支持拼接一段文字，比如 ```python ，最终python并不会展示到代码块中。

## 主要规则
设计以下规则：
1. 每个代码块有各种不同的标签开头，这个标签后面的同一行内容都会作为规则或者未来预留规则的一部分
2. 每个标签都需要定义一个id，同一个标签下，不能有重复的id（可以没有id，只有需要配对的，比如#test和#test-result）
3. 测试代码块用 #test 开头作为标签
4. 测试的预期输出用 #test-result 开头作为标签
5. 每个测试代码块（#test标签）如果要被测试到，需要有一个id相同的对应的#test-result标签
6. 如果只是一个不需要测试，但需要执行的代码块，称为加载代码块，用 #test-setup 开头作为标签
7. 每个测试代码块需要指定语言，目前支持 `shell` 和 `python`（`#test-result` 不需要）
8. 标签里的不同参数，用空格分隔
9. 整个代码块的加载顺序是按照文档的从前到后的顺序，无标签的代码块会跳过。
10. 用\<!-- -->注释可以用来加载一些隐藏的#test-setup代码块，也只能加载#test-setup

### 样例

下面用一个最小可跑的 quick-start 片段，把 10 条主要规则都演示到。

#### 规则 1 / 3 / 4 / 7 / 8：基本配对，指定语言（#test-result 不需要），多个空格分隔参数

````markdown
```python #test id="somke"
print(1 + 2)
print(2 + 2)
```

```python #test-result id="somke" fuzzy='xxx'
3
xxx
```
````

`#test` 块用 `id="somke"` 标记，`#test-result` 块用同 `id` 配对；`python` / `shell` 是 `#test` 块必填的语言前缀（规则 7），`#test-result` **不需要**语言前缀（也是规则 7）。`id="somke" fuzzy='xxx'` 这两个参数用空格分隔（规则 8）。

#### 规则 5：id 配对

````markdown
```shell #test id="pwd"
pwd
```

```shell #test-result id="pwd"
/hdc
```
````

`#test id="pwd"` 必须有**一个** `id="pwd"` 的 `#test-result` 配对；缺一个或多个都让 runner 报告"未配对 / 重复配对"并整体失败。

#### 规则 2：同标签下 id 必须唯一

````markdown
```shell #test id="dup"
echo a
```

```shell #test id="dup"
echo b
```

```shell #test-result id="dup"
b
```
````

同一 `#test` 标签下出现重复 `id="dup"` 是文档错误，runner 报告"重复 id"并整体失败；其他标签（`#test-result` / `#test-setup`）的 `id` 也各自独立唯一——`#test` 与 `#test-result` 的 `id` 在另一维度上必须配对相等。

#### 规则 6：加载代码块 #test-setup

````markdown
```shell #test-setup store="today"
date +%Y-%m-%d
```
# stdout: 2026-08-20
````

`#test-setup` 它的 stdout 全部由 `store='today'` 捕获，stderr 与退出码不参与（脚本非零退出让整个 doc fail）。

#### 规则 9：按文档顺序执行，无标签代码块被跳过

````markdown
```shell
# 无标签的代码块
echo "this block is skipped"
```

```shell #test id="after-skip"
echo "after skip"
```

```shell #test-result id="after-skip"
after skip
```
````

无任何标签前缀的代码块被 runner 完全跳过——既不执行也不参与测试。runner 按文档从前到后只处理有标签的块，块之间的相对顺序就是执行顺序；本例里 `#test id="after-skip"` 紧跟一个被跳过的块，runner 仍然只看到 `after-skip` 并执行它。

#### 规则 10：用 HTML 注释隐藏 #test-setup

````markdown
<!-- 这是一个隐藏的环境初始化块，markdown 渲染器会丢掉注释里的全部内容，读者看不到 
```shell #test-setup store="home"
echo $HOME
```
-->
````

HTML 注释 `<!-- ... -->` 可用来包裹 `#test-setup` 块，让它**不渲染到页面上**——markdown 渲染器直接丢掉注释内容，读者看不到这段代码；但 runner 读的是源文件，仍会执行并 `store`。这条规则**只对 `#test-setup` 有效**：`#test` / `#test-result` 是给读者抄、给读者看的，注释隐藏它们会让 runner 也看不到对应块，整个 doc 直接 fail。

## 测试结果 #test-result的扩展规则
1. 默认模糊匹配的内容是 '...' 会被解析为正则的非贪婪匹配，支持多行
2. 如果要指定被模糊的字符，使用 fuzzy='xxx' ，这里 xxx 就会被替换成非贪婪匹配，可以指定多个，当指定了fuzzy，默认的'...'会被当作普通字符
3. 可以使用disable_fuzzy来取消所有的非贪婪匹配，包括默认的，配置了fuzzy，会报错

### 样例

#### 默认 `...`

````markdown
```shell #test id="hello-default"
echo "hello"
echo "world"
```

```shell #test-result id="hello-default"
hello
...
```
````

`...` 作为占位符匹配任意内容（正则非贪婪，支持跨行）；上面例子匹配 `world` 的一行。

#### `fuzzy='xxx'`

````markdown
```shell #test id="py-version"
python --version
```

```shell #test-result id="py-version" fuzzy='xxx'
Python 3.xxx
```
````

当默认 `...` 与文档中其他含义冲突，可用 `fuzzy='xxx'` 改用其他占位符；`xxx` 在本例里被解析为**非贪婪匹配**，可覆盖 `3.12.5`、`3.11.10` 等输出。

#### `disable_fuzzy`

````markdown
```shell #test id="literal-dots"
echo "output: ... here"
```

```shell #test-result id="literal-dots" disable_fuzzy
output: ... here
```
````

`disable_fuzzy` 是**无值参数**（直接写名字，不带 `=`）：所有 placeholder（包括默认的 `...` 和 `fuzzy=` 声明过的）都被视为字面字符匹配。本例中 expected 里的 `...` 不再是非贪婪通配，而是必须按字面出现的三个点。

**互斥约束**：和 `fuzzy=` 一起写会直接报错 `disable_fuzzy conflicts with fuzzy=`：

````markdown
```shell #test-result id="bad" fuzzy='xxx' disable_fuzzy
...
```
````

这条会抛 `LabelSpecError`，所以两者只能选一个。

## 加载代码块 #test-setup的扩展规则
1. 支持将代码块执行的输出到变量里为后续使用，使用 store='xxx' , xxx 作为变量名，可以被后续其他标签引用，如果重复设置，后面的store='xxx'会覆盖前面的。其中代码块执行失败的输出不会被store。

### 样例

````markdown
```shell #test-setup store="today"
date +%Y-%m-%d
```

```shell #test-setup store="today"
date +%H:%M:%S
```
````

执行后 `today` 被存为 `2026-08-20`，然后**被第二次 `store='today'` 覆盖为 `14:23:45`**——后面的 `#test` 块引用到的是覆盖后的值。stderr 与退出码不参与（失败退出让整个 doc fail）。

## 获取被#test-setup store的变量
1. 在标签里使用 load='xxx>>yyy' ，其中xxx是当前代码块之前历史用store='xxx' 存储过的xxx，yyy是当前代码块里希望被加载进去的，以<yyy>的格式。如果有多个，可以写多次load='xxx>>yyy'

### 样例

````markdown
```shell #test-setup store="user"
whoami
```
# 比如这里whoami输出：hdc

```shell #test id="greet" load="user>>name"
echo "hello <name>"
```

```shell #test-result id="greet"
hello hdc
```
````

`xxx>>yyy` 中 `xxx` 是 `store` 时用的变量名，`yyy` 是当前块里**新**的占位符名。`<name>` 在命令执行和预期输出比对前都会被 `load` 替换为 `store` 捕获的实际值（如 `hdc`）。

#### 多个 `load`

````markdown
```shell #test-setup store="host"
hostname
```
# 这里输出：host

```shell #test-setup store="user"
whoami
```
# 这里输出：hdc

```shell #test id="prompt" load="host>>h" load="user>>u"
echo "<u>@<h>"
```

```shell #test-result id="prompt"
hdc@host
```
````

多次 `load` 用空格分隔，分别注入不同占位符；`<h>` / `<u>` 在命令与预期输出里都会被替换为实际值。