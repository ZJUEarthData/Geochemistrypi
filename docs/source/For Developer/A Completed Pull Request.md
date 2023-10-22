#  A Completed Pull Request

Takeaway:

+ Virtual Environment

+ Branch Operation
+ Continuous Integration

1. Search how to download **git** as a command line tool in your computer.

2. Find the codebase url through the green button on our GitHub repository.

    <img width="1174" alt="image" src="https://user-images.githubusercontent.com/47497750/236991014-7c17248c-54cc-41f0-af15-796a67de5307.png">

3. Use `git clone <url>` to clone the remote codebase into your local machine through the command line. No worry if your shell (command line) looks different from mine (The operations are same in the Mac and Windows systems).

    <img width="920" alt="image" src="https://user-images.githubusercontent.com/47497750/236991322-843c9497-3c32-42d0-a564-1979e093a8d1.png">

4. Switch into **geochemistrypi** repository. Activate your virtual environment (**venv** or **conda**). In this case, I use conda to set up.  It is a good habit to use a specific virtual environment for a specific Python project. Then, use `pip install -r requirements.txt` to download the dependencies.

    <img width="921" alt="image" src="https://user-images.githubusercontent.com/47497750/237021355-0fff7b74-ebd8-47f6-8c76-8918b11fc8ec.png">

    <img width="921" alt="image" src="https://user-images.githubusercontent.com/66153455/262966478-311bb75e-e149-410b-9074-7aba7b4bbb03.jpg">

5. Use `pre-commit install` to set up the git hook scripts, which will be used later on, no need to worry! You only need to run this  command once in this local repository unless you clone another one again in other local directory.

    <img width="920" alt="image" src="https://user-images.githubusercontent.com/47497750/237022518-7e0dfa23-7b49-46f3-adf2-ec1f83651f44.png">

    <img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/262966493-d147a599-4b18-4974-80e3-00b1f620c96b.jpg">

6. Use **Visual Studio Code** (Recommended IDE) to open the codebase and don't forget to activate your virtual environment as the picture shows below.

    <img width="1439" alt="image" src="https://user-images.githubusercontent.com/47497750/237025973-aa9792a3-dbfa-41f3-8adf-3d5824146ddf.png">

7. Create a new git branch to develop on your own path. You can name the branch name as your identification or as the function you are going to build up.

    <img width="1219" alt="image" src="https://user-images.githubusercontent.com/47497750/237028303-e3edf748-2f76-40a3-8247-3869656dfc88.png">

    <img width="1219" alt="image" src="https://user-images.githubusercontent.com/66153455/262966498-aaa905ab-63a5-4231-845d-b217180b998f.jpg">

8. Do some modifications on the codebase. For example, in this case, I update the badge information of READM.md.

    <img width="1433" alt="image" src="https://user-images.githubusercontent.com/47497750/237028688-82e628f8-eaac-4cf2-a3a4-c2eb51a85cd8.png">

9. After finishing modifications on the codebase, you need to use `git status` to check what file has been modified.

    <img width="1438" alt="image" src="https://user-images.githubusercontent.com/47497750/237053725-303b6c15-5939-441e-8bee-2d322c2e8019.png">

10. It is a good habit to use `git diff <filename>` check whether the changed content on that specific files respectively is what you desire to make. After enter the command, the console will switch into **diff display**. In  **diff displays** , press the keybord `j` to scroll down and press the keyboard `k`  to scroll up. `+` refers to the line added while `-` refers to the line deleted. Press the keyboard `q` to exit from the **diff display** after checking.

    <img width="1200" alt="image" src="https://user-images.githubusercontent.com/47497750/237054080-39198eac-e804-41db-b672-aece0417fe38.png">

    <img width="1274" alt="image" src="https://user-images.githubusercontent.com/47497750/237013451-f7bd73f8-6b3f-4781-8954-5aaf1e171f1f.png">

11. Use `git add <filename>` to staged the files for commit.

    <img width="1202" alt="image" src="https://user-images.githubusercontent.com/47497750/237054509-e8418115-23c3-49c7-8690-51dc31d2aed1.png">

12. Use `git commit -m "tag: message"` to make a local commit. It is a good habit to follow up the principle that one commit for one functionality implementation or one optimization or one bug fix. It is allowed to push multiple commits to the remote codebase with one pull request.

    Please include the following tags in the beginning of your commit message to make more organized commits and PRs. It would tell exactly what use it is in this commit.

    + `feat`: a new feature is introduced with the changes
    + `fix`: a bug fix has occurred
    + `perf`: an existing feature improved
    + `docs`: changes to the documentation
    + `style`: code formatting
    + `refactor`: refactoring production code
    + `revert`:  version revertion
    + `chore`: the change of developing tools or assisted tool
    + `test`: adding missing tests, refactoring tests
    + `build`: package the codebase
    + `ci`: continue integration
    + `BREAKING CHANGE`: a breaking API change

    <img width="1204" alt="image" src="https://user-images.githubusercontent.com/47497750/237055438-38e60ffc-a54d-4a25-82d3-d5e8b33357d6.png">

13. Continuous Integration (CI): In our codebase, we set up a pre-commit locally before.  Normally, it will automatically check code grammar and correct code styles for your commited code.

    (1) In this **successful case**, it indicates that everything works well. No code grammar mistake and code style conforms to the configuration. Hence, no need to do further, you can push the local commit to remote codebase in the GitHub.

    <img width="1202" alt="image" src="https://user-images.githubusercontent.com/47497750/237055329-cb3935b8-44d7-4205-87f3-4ae4dbcbeca6.png">

    (2) In another **failed case** (I did another experiment before, not in this docs!), it indicates that there are three files where import statements don't conform to the specification. Hence, it corrects them for you, which means that you don't need to manually correct them by yourself.

    <img width="1160" alt="Pasted Graphic 9" src="https://user-images.githubusercontent.com/47497750/237057185-f43b9867-7f42-4505-a90e-0c954923a1da.png">

    The next step is to use `git status`  and `git diff` to check the details in the corrected files to see what CI tool has done for you.

    <img width="627" alt="Pasted Graphic 13" src="https://user-images.githubusercontent.com/47497750/237058359-6616e9aa-ad44-488f-bfea-b290a6c8cb08.png">

    <img width="576" alt="Pasted Graphic 14" src="https://user-images.githubusercontent.com/47497750/237058693-95e9b454-92be-4e47-a7ab-9031c7645849.png">

    You would see that you need to use `git add` again. Because CI tool has modified your code , you need to do a standard git commit process again.

    <img width="800" alt="Pasted Graphic 16" src="https://user-images.githubusercontent.com/47497750/237059215-08ef553a-be5e-46bf-9c5c-e76342773d8b.png">

    Again, you need to run `git commit` again. Just use the previous same command. Now, it passes! Fancy is it? Until now, you are allowed to push the local commit to remote codebase in the GitHub.

    <img width="1109" alt="Pasted Graphic 17" src="https://user-images.githubusercontent.com/47497750/237059295-42b855e9-f70a-4894-8e34-0b406e1d9f53.png">

14. Back to the **successful case**. Now, it is time to push your local  **dev/Sany**  branch to the remote  **dev/Sany** branch. It may not created yet, but no worried, it will be shown later. Firstly, use `git pull origin main --rebase`. This command pulls the latest changes from the remote **main** branch and applies them to your local **dev/Sany** branch, while also rebasing your local changes on top of the new changes from the remote.

    (1) **No conflict case**: In this example, it indicates that there is no conflict with the remote **main** branch. Quite normally, sometimes, when you are coding in your own branch, the remote **main** branch is far ahead of yours. It is likely that there are conflicts existing so you need to resolve the conflicts before merge your remote **dev/Sany** branch into remote **main** branch.

    The way through `git pull origin main --rebase` can save you troubles. If there are any conflicts between your local  **dev/Sany** changes and the remote **main** branch's changes, Git will stop the rebase process and alert you that there are conflicts that need to be resolved. In this case, there is no conflict at all.

    <img width="1200" alt="image" src="https://user-images.githubusercontent.com/47497750/237062092-0a922f60-4485-4343-93ca-89ed8902acd6.png">

    If in your own case, there are some conflicts, to resolve the conflicts, you will need to edit the affected files manually on VSCode IDE to resolve the differences between the conflicting changes. Once you have made the necessary changes, you can stage the changes using `git add`, and then use `git rebase --continue` to continue the rebase process.

    Git will then apply your local changes on top of the new changes from the remote, incorporating the changes you made to resolve the conflicts. It's worth noting that resolving conflicts can sometimes be a complex process, especially when there are multiple conflicts across several files. If you're not familiar with resolving conflicts in Git, it's a good idea to read up on it first or seek help from experienced Git users (Sany or others) to avoid any unintended consequences.

    The picture below is the successful case without conflicts.

    <img width="1217" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/ae7de931-dc30-45d5-a29a-0389152075c7">

    After resolving all conflicts, you can then use `git push origin dev/Sany:dev/Sany` to push your local changes to the remote repository. This command pushes your local **dev/Sany** branch to the remote repository and creates a new remote **dev/Sany** branch if it does not exist yet. The colon (`:`) separates the local branch name from the remote branch name.

    <img width="1222" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/3ba452c0-c93a-430c-9652-810babd00236">

    （2）**Unstaged changes case**: In this example, it indicates that some files, which are modified and tracked by git, haven't been commited yet. Probably, these files are not the ones you want to push because they are under development. In Git's protection mechanism, when there is some tracked and modified files in your working directory, you are not allowed to pull and merge the remote branch **main** into your local branch. However, if your working directory only includes untracked files, this problem like the picture shown below wouldn' t happend.

    <img width="947" alt="Pasted Graphic 8" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/387b4870-f5b1-4d4d-9865-4564c3b8e5c3">

    Fortunately, you can use `git stash push` to save those changed files into a hidden zone temporarily without committing them.

    <img width="1012" alt="Pasted Graphic 3" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/00c628b6-a453-490c-af77-c37929b65a13">

    Now, you can use `git pull origin main --rebase` and follow the same procedure as in the **no conflict case**.

    <img width="1005" alt="Pasted Graphic 7" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/82a5fb49-022f-43f9-89b8-862c237de2ba">

    <img width="816" alt="Pasted Graphic 5" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/c286eb02-91e4-4927-8b69-981b6aa7a902">

    Once you finish pushing, you can use `git stash pop` to bring back the modified files into your working directory. For more information about **git stash** , plz google it.

    <img width="816" alt="Pasted Graphic 6" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/6472df94-8d65-45aa-b787-1848966fe90d">

15. Now, it is time to creat your first **PULL REQUEST**. Wowww, what a long journey! Copy the url with underline https://github.com/ZJUEarthData/geochemistrypi/pull/new/dev/Sany (it is different in your case, check your own information) and open it with a browser. You would the page directing you to create a **pull request** through clicking the green button. It is done!!! Then, no need to operate anymore . As a normal developer, you are not allowed to merge by yourself. Our maintainers will check and merge later on if it is good.

    <img width="1041" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/7cd4a746-2b78-4861-a76d-758c7ab9e21a">

    <img width="1193" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/d9ea1bf1-c452-4923-ac76-138a116383fe">

    The page below is only visible to our maintainers.

    <img width="1185" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/d63e4339-e465-4b79-adb3-ae5a057c7433">
